# =====================================================================
# check_sev_dist.py -- Modélisation du Coût de Sévérité (Gamma vs Log-Normale)
# =====================================================================

import sys
from pathlib import Path

# Importer la configuration globale et ses utilitaires
sys.path.append(str(Path(__file__).resolve().parent))
from config import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

print("\n==========  MODÉLISATION DU COÛT DE SÉVÉRITÉ ==========\n")

# --- 1) Chargement & Filtrage avec SEUIL_GRAVE (10 000 €) --------------
df_train = load_pkl("df_train")
df_val   = load_pkl("df_val")

# Sélection des sinistres réels payés
df_train_sev = df_train[(df_train["ChargeTot"] > 0) & (df_train["NbSinSev"] > 0)].copy()

# Calcul du coût moyen unitaire exact par sinistre payé
df_train_sev["AvgClaim"] = df_train_sev["ChargeTot"] / df_train_sev["NbSinSev"]

# Écrêtement de la sévérité au seuil métier (10 000 EUR)
df_train_sev["AvgClaim_Capped"] = np.minimum(df_train_sev["AvgClaim"], SEUIL_GRAVE)

# DEFINITION DE ATTR_X POUR LES TESTS ET GRAPHIQUES
attr_x = df_train_sev["AvgClaim_Capped"].values
n_claims = len(attr_x)  # Nombre de sinistres analysés

print(f"Nombre de sinistres analysés : {len(df_train_sev):,}".replace(",", " "))
print(f"Seuil d'écrêtement appliqué (SEUIL_GRAVE) : {SEUIL_GRAVE:,.0f} EUR".replace(",", " "))
print(f"Coût moyen brut : {df_train_sev['AvgClaim'].mean():.2f} EUR")
print(f"Coût moyen attritionnel (écrêté) : {df_train_sev['AvgClaim_Capped'].mean():.2f} EUR")

# A) Gamma (Ajustement paramétrique)
shape_g, loc_g, scale_g = stats.gamma.fit(attr_x, floc=0)

# B) Log-Normale (Ajustement paramétrique)
shape_ln, loc_ln, scale_ln = stats.lognorm.fit(attr_x, floc=0)

# --- 3) Formule et Ajustement du GLM Gamma sur AvgClaim_Capped --------
formula_sev = (
    "AvgClaim_Capped ~ C(DrivAgeBand) + C(VehAgeBand) + C(VehPowerBand) + "
    "C(BonusMalusBand) + C(DensityBand) + C(VehGas) + C(Area)"
)

print("Ajustement du GLM Gamma sur la sévérité attritionnelle écrêtée (<= 10 000 €)...")
model_gamma = smf.glm(
    formula=formula_sev,
    data=df_train_sev,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    freq_weights=df_train_sev["NbSinSev"]
).fit()

# --- Ajustement du GLM Log-Normal (sur sévérité écrêtée) -------------
print("Ajustement du GLM Log-Normal (Gaussian sur log de la sévérité écrêtée)...")

# 1. Calcul du log sur la variable écrêtée (<= 10 000 €)
df_train_sev["log_AvgClaim_Capped"] = np.log(df_train_sev["AvgClaim_Capped"])

# 2. Formule explicite ciblée sur la nouvelle variable
formula_lognormal = (
    "log_AvgClaim_Capped ~ C(DrivAgeBand) + C(VehAgeBand) + C(VehPowerBand) + "
    "C(BonusMalusBand) + C(DensityBand) + C(VehGas) + C(Area)"
)

# 3. Ajustement du modèle Gaussien (Log-Normal)
model_lognormal = smf.glm(
    formula=formula_lognormal,
    data=df_train_sev,
    family=sm.families.Gaussian(link=sm.families.links.Identity()),
    freq_weights=df_train_sev["NbSinSev"]  # Alignement sur le nombre de sinistres payés
).fit()

# --- 4) Tableau Comparatif Corrige (Ajustement Jacobien) --------------
# Utilisation de 'log_AvgClaim_Capped' au lieu de 'log_AvgClaim'
sum_log_y = float(np.sum(df_train_sev["log_AvgClaim_Capped"]))

# Correction Jacobienne pour la Log-Normale sur l'échelle des Euros
llf_ln_adj = float(model_lognormal.llf) - sum_log_y
aic_ln_adj = -2 * llf_ln_adj + 2 * (model_lognormal.df_model + 2)
bic_ln_adj = -2 * llf_ln_adj + np.log(n_claims) * (model_lognormal.df_model + 2)

df_comp_sev = pd.DataFrame([
    {
        "Loi": "Gamma (Lien Log)",
        "Log-Likelihood": float(model_gamma.llf),
        "AIC": float(model_gamma.aic),
        "BIC": float(model_gamma.bic_llf),
        "Déviance": float(model_gamma.deviance)
    },
    {
        "Loi": "Log-Normale (Ajustée EUR)",
        "Log-Likelihood": llf_ln_adj,
        "AIC": aic_ln_adj,
        "BIC": bic_ln_adj,
        "Déviance": float(model_lognormal.deviance)
    }
])

print("\n" + "="*85)
print("     TABLEAU COMPARATIF CORRIGÉ : SÉVÉRITÉ GAMMA vs LOG-NORMALE (ÉCHELLE EUR)     ")
print("="*85)
print(f"{'Loi':<25} | {'AIC':<10} | {'BIC':<10} | {'Déviance':<10}")
print("-" * 85)
for _, row in df_comp_sev.iterrows():
    print(f"{row['Loi']:<25} | {row['AIC']:>10,.1f} | {row['BIC']:>10,.1f} | {row['Déviance']:>10,.1f} ".replace(",", " "))
print("="*85 + "\n")

# --- 5) Diagnostic des Résidus (sur la cible écrêtée) -----------------
print("Génération du diagnostic des résidus pour le modèle Gamma (écrêté)...")

# Calcul des résidus par rapport à AvgClaim_Capped
res_bruts_gamma = df_train_sev["AvgClaim_Capped"].values - model_gamma.fittedvalues.values
res_pearson_gamma = model_gamma.resid_pearson.values
obs_index = np.arange(len(df_train_sev))

fig_res_gamma, (ax1_res, ax2_res) = plt.subplots(1, 2, figsize=(12, 5))

# A) Résidus de Pearson
ax1_res.scatter(obs_index, res_pearson_gamma, color="#003399", s=5, alpha=0.7)
ax1_res.set_xlabel("Observations")
ax1_res.set_ylabel("pearson residual (Gamma)")
ax1_res.grid(False)

# B) Résidus Bruts
ax2_res.scatter(obs_index, res_bruts_gamma, color="#003399", s=5, alpha=0.7)
ax2_res.set_xlabel("Observations")
ax2_res.set_ylabel("Raw residual (Gamma)")
ax2_res.grid(False)

plt.tight_layout()
save_fig(fig_res_gamma, "diagnostics_residus_gamma")

# --- 6) Graphique 2 : QQ-Plot avec sous-échantillonnage --------------
np.random.seed(42)
sample_size = min(3000, n_claims)
idx_sample = np.random.choice(n_claims, size=sample_size, replace=False)
emp_sample = np.sort(attr_x[idx_sample])  # attr_x est déjâ écrêté à SEUIL_GRAVE

# Plafond d'affichage calé sur SEUIL_GRAVE (10 000 EUR)
xcap = float(SEUIL_GRAVE)

pp = np.linspace(1 / (sample_size + 1), sample_size / (sample_size + 1), sample_size)
theo_gamma = stats.gamma.ppf(pp, shape_g, loc_g, scale_g)
theo_lnorm = stats.lognorm.ppf(pp, shape_ln, loc_ln, scale_ln)

fig_qq, (ax1_qq, ax2_qq) = plt.subplots(1, 2, figsize=(12, 5))

# A) QQ-Plot Gamma
ax1_qq.scatter(theo_gamma, emp_sample, color="#003399", s=10, alpha=0.6)
ax1_qq.plot([0, xcap], [0, xcap], color="red", linestyle="--", label="y = x")
ax1_qq.set_xlim(0, xcap)
ax1_qq.set_ylim(0, xcap)
ax1_qq.set_title("QQ-plot : Loi Gamma (écrêtée à 10k €)")
ax1_qq.set_xlabel("Quantiles théoriques (EUR)")
ax1_qq.set_ylabel("Quantiles empiriques (EUR)")
ax1_qq.grid(False)

# B) QQ-Plot Log-Normale
ax2_qq.scatter(theo_lnorm, emp_sample, color="#003399", s=10, alpha=0.6)
ax2_qq.plot([0, xcap], [0, xcap], color="red", linestyle="--", label="y = x")
ax2_qq.set_xlim(0, xcap)
ax2_qq.set_ylim(0, xcap)
ax2_qq.set_title("QQ-plot : Loi Log-Normale (écrêtée à 10k €)")
ax2_qq.set_xlabel("Quantiles théoriques (EUR)")
ax2_qq.set_ylabel("Quantiles empiriques (EUR)")
ax2_qq.grid(False)

plt.tight_layout()
save_fig(fig_qq, "attritionnels_qqplot")

# --- Diagnostic des Résidus pour le modèle retenu (Log-Normal) --------
print("Génération du diagnostic des résidus pour le modèle Log-Normal...")

# 1. Estimation de la variance résiduelle sur l'échelle logarithmique (sigma^2)
sigma2_hat = model_lognormal.scale

# 2. Conversion des prédictions en EUR avec correction de Smearing (Inégalité de Jensen)
pred_ln_eur = np.exp(model_lognormal.fittedvalues + sigma2_hat / 2.0)

# 3. Calcul des résidus :
# - Pearson : calculés directement sur l'échelle log par statsmodels
# - Bruts : écarts réels en Euros sur la cible écrêtée
res_pearson_ln = model_lognormal.resid_pearson.values
res_bruts_ln = df_train_sev["AvgClaim_Capped"].values - pred_ln_eur
obs_index = np.arange(len(df_train_sev))

# 4. Construction des graphiques
fig_res_ln, (ax1_res, ax2_res) = plt.subplots(1, 2, figsize=(12, 5))

# A) Résidus de Pearson (Échelle Log)
ax1_res.scatter(obs_index, res_pearson_ln, color="#003399", s=5, alpha=0.7)
ax1_res.axhline(0, color="red", linestyle="--", linewidth=1)
ax1_res.set_xlabel("Observations")
ax1_res.set_ylabel("Résidus de Pearson (Log)")
ax1_res.set_title("Résidus de Pearson (Log-Normale)")
ax1_res.grid(False)

# B) Résidus Bruts (en Euros)
ax2_res.scatter(obs_index, res_bruts_ln, color="#003399", s=5, alpha=0.7)
ax2_res.axhline(0, color="red", linestyle="--", linewidth=1)
ax2_res.set_xlabel("Observations")
ax2_res.set_ylabel("Résidus Bruts (EUR)")
ax2_res.set_title("Résidus Bruts (Cible écrêtée à 10k €)")
ax2_res.grid(False)

plt.tight_layout()
save_fig(fig_res_ln, "diagnostics_residus_lognormale")

# --- 8) Application des Prédictions & Sauvegardes (Log-Normale) ------
print("\n========== SAUVEGARDE DES PRÉDICTIONS ET DU MODÈLE ==========\n")

# 1. Récupération de la variance résiduelle sigma^2 sur l'échelle log
sigma2_hat = float(model_lognormal.scale)

# 2. Fonction de prédiction monétaire avec correction de biais
def predict_sev_lognormal(model, df_target, sigma2):
    log_pred = model.predict(df_target)
    return np.exp(log_pred + sigma2 / 2.0)

# 3. Calcul de la sévérité prédite en EUR sur l'ensemble du portefeuille
df_train["Pred_Sev_LogNormal"] = predict_sev_lognormal(model_lognormal, df_train, sigma2_hat)
df_val["Pred_Sev_LogNormal"]   = predict_sev_lognormal(model_lognormal, df_val, sigma2_hat)


df_train["Pred_Sev"] = df_train["Pred_Sev_LogNormal"]
df_val["Pred_Sev"]   = df_val["Pred_Sev_LogNormal"]

print(f"Sévérité moyenne prédite (Train) : {df_train['Pred_Sev_LogNormal'].mean():.2f} EUR")
print(f"Sévérité moyenne prédite (Val)   : {df_val['Pred_Sev_LogNormal'].mean():.2f} EUR")

# 4. Sauvegarde des bases enrichies
save_pkl(df_train, "df_train")
save_pkl(df_val, "df_val")

# 5. Sauvegarde du modèle Log-Normal
save_pkl(model_lognormal, "model_lognormal_glm")

print(" OK - Analyse du coût de sévérité terminée. Fichiers et graphiques enregistrés dans output/.\n")
# =====================================================================
# check_freq_dist.py -- Test des Lois de Fréquence (Poisson, NB, ZIP)
# Projet : Tarification Automobile Non-Vie (GLM vs XGBoost)
# =====================================================================

import sys
from pathlib import Path

# Importer la configuration globale et ses utilitaires
sys.path.append(str(Path(__file__).resolve().parent))
from config import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, nbinom

print("\n========== - TEST DES LOIS DE FRÉQUENCE (POISSON, NB, ZIP ) ==========\n")

# --- 1) Chargement des données & Sécurisation des tranches -----------
df_train = load_pkl("df_train")
df_val   = load_pkl("df_val")

# Vérification / Création des tranches tarifaires si absentes
for df in [df_train, df_val]:
    if "DrivAgeBand" not in df.columns:
        df["DrivAgeBand"] = pd.cut(df["DrivAge"], bins=[17, 22, 26, 42, 74, 110], labels=["18-22", "23-26", "27-42", "43-74", "75+"])
    if "VehAgeBand" not in df.columns:
        df["VehAgeBand"] = pd.cut(df["VehAge"], bins=[-1, 0, 5, 12, 110], labels=["0", "1-5", "6-12", "13+"])
    if "VehPowerBand" not in df.columns:
        df["VehPowerBand"] = pd.cut(df["VehPower"], bins=[0, 5, 7, 9, 110], labels=["<=5", "6-7", "8-9", "10+"])
    if "BonusMalusBand" not in df.columns:
        df["BonusMalusBand"] = pd.cut(df["BonusMalus"], bins=[0, 50, 60, 90, 350], labels=["50", "51-60", "61-90", "91+"])
    if "DensityBand" not in df.columns:
        df["DensityBand"] = pd.qcut(df["Density"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

formula_glm = (
    "ClaimNb ~ C(DrivAgeBand) + C(VehAgeBand) + C(VehPowerBand) + "
    "C(BonusMalusBand) + C(DensityBand) + C(VehGas) + C(Area)"
)

# --- 2) Ajustement des 3 Modèles -------------------------------------
print("1. Ajustement du GLM Poisson...")
model_poisson = smf.glm(
    formula=formula_glm,
    data=df_train,
    family=sm.families.Poisson(),
    offset=np.log(df_train["Exposure"])
).fit()

print("2. Ajustement du GLM Binomiale Négative (NB2)...")
model_nb = smf.glm(
    formula=formula_glm,
    data=df_train,
    family=sm.families.NegativeBinomial(),
    offset=np.log(df_train["Exposure"])
).fit()

print("3. Ajustement du modèle Zero-Inflated Poisson (ZIP)...")
model_zip = sm.ZeroInflatedPoisson.from_formula(
    formula=formula_glm,
    data=df_train,
    offset=np.log(df_train["Exposure"]),
    exog_infl=np.ones((len(df_train), 1))
).fit(method='bfgs', maxiter=100, disp=False)


# --- 3) Tableau Récapitulatif de Comparaison -------------------------
def get_metrics(model, name):
    ll = float(model.llf)
    aic = float(model.aic)
    bic = float(model.bic if hasattr(model, 'bic') else getattr(model, 'bic_llf', -2 * ll))
    dev = float(getattr(model, 'deviance', -2 * ll))
    return {"Modèle": name, "Log-Likelihood": ll, "AIC": aic, "BIC": bic, "Déviance": dev}

df_recap = pd.DataFrame([
    get_metrics(model_poisson, "Poisson"),
    get_metrics(model_nb, "Binomiale Négative (NB)"),
    get_metrics(model_zip, "Zero-Inflated Poisson (ZIP)")
])

# Affichage récapitulatif
print("\n" + "="*85)
print("          TABLEAU RÉCAPITULATIF DE COMPARAISON DES 4 LOIS DE FRÉQUENCE          ")
print("="*85)
print(f"{'Modèle':<28} | {'Log-Likelihood':<15} | {'AIC':<12} | {'BIC':<12} | {'Déviance':<12}")
print("-" * 85)
for _, row in df_recap.iterrows():
    print(f"{row['Modèle']:<28} | {row['Log-Likelihood']:>15,.2f} | {row['AIC']:>12,.2f} | {row['BIC']:>12,.2f} | {row['Déviance']:>12,.2f}".replace(",", " "))
print("="*85 + "\n")

save_tab(df_recap, "recapitulatif_4_lois_frequence")

# --- 4) Graphique de Distribution : Poisson vs Binomiale Négative ----
k_vals = np.array([0, 1, 2, 3, 4])
obs_counts = df_train["ClaimNb"].value_counts().sort_index().reindex(k_vals, fill_value=0).values

# Prédictions d'espérances
mu_poisson = model_poisson.fittedvalues.values
mu_nb = model_nb.fittedvalues.values
alpha_nb = model_nb.scale  # Paramètre de dispersion alpha

exp_poisson = [np.sum(poisson.pmf(k, mu_poisson)) for k in k_vals]
exp_nb = [np.sum(nbinom.pmf(k, n=1/alpha_nb, p=1/(1 + alpha_nb * mu_nb))) for k in k_vals]

fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
x = np.arange(len(k_vals))
width = 0.25

ax_dist.bar(x - width, obs_counts, width, label="Observé", color="#1f4e78")
ax_dist.bar(x, exp_poisson, width, label="Poisson Théorique", color="#d9534f", alpha=0.85)
ax_dist.bar(x + width, exp_nb, width, label="Binomiale Négative (NB)", color="#2e7d32", alpha=0.85)

ax_dist.set_yscale("log")
ax_dist.set_xticks(x)
ax_dist.set_xticklabels(k_vals)
ax_dist.set_xlabel("Nombre de sinistres (k)")
ax_dist.set_ylabel("Effectif de contrats (échelle log10)")
ax_dist.set_title("Ajustement de la fréquence des sinistres : Poisson vs Binomiale Négative")
ax_dist.legend()

for i in range(len(k_vals)):
    ax_dist.text(i - width, obs_counts[i] * 1.15, f"{obs_counts[i]:,}".replace(",", " "), ha='center', fontsize=7)
    ax_dist.text(i, exp_poisson[i] * 1.15, f"{int(exp_poisson[i]):,}".replace(",", " "), ha='center', fontsize=7)
    ax_dist.text(i + width, exp_nb[i] * 1.15, f"{int(exp_nb[i]):,}".replace(",", " "), ha='center', fontsize=8, fontweight='bold', color='#1b5e20')

plt.tight_layout()
save_fig(fig_dist, "ajustement_poisson_vs_nb")

# --- 5) Graphiques Diagnostics des Résidus (Poisson & NB) -------------
obs_index = np.arange(len(df_train))

# A) Résidus Poisson
fig_res_p, (ax1_p, ax2_p) = plt.subplots(1, 2, figsize=(12, 5))
ax1_p.scatter(obs_index, model_poisson.resid_pearson.values, color="#003399", s=5, alpha=0.7)
ax1_p.set_xlabel("Observations")
ax1_p.set_ylabel("pearson residual (Poisson)")
ax1_p.grid(False)

ax2_p.scatter(obs_index, model_poisson.resid_response.values, color="#003399", s=5, alpha=0.7)
ax2_p.set_xlabel("Observations")
ax2_p.set_ylabel("Raw residual (Poisson)")
ax2_p.grid(False)

plt.tight_layout()
save_fig(fig_res_p, "diagnostics_residus_poisson")

# B) Résidus NB
fig_res_nb, (ax1_nb, ax2_nb) = plt.subplots(1, 2, figsize=(12, 5))
ax1_nb.scatter(obs_index, model_nb.resid_pearson.values, color="#003399", s=5, alpha=0.7)
ax1_nb.set_xlabel("Observations")
ax1_nb.set_ylabel("pearson residual (NB)")
ax1_nb.grid(False)

ax2_nb.scatter(obs_index, model_nb.resid_response.values, color="#003399", s=5, alpha=0.7)
ax2_nb.set_xlabel("Observations")
ax2_nb.set_ylabel("Raw residual (NB)")
ax2_nb.grid(False)

plt.tight_layout()
save_fig(fig_res_nb, "diagnostics_residus_nb")

print("\n OK - Test des 3 lois (Poisson, NB, ZIP). \n")
# =====================================================================
# SAUVEGARDE DES RÉSULTATS POUR LE CALCUL DE LA PRIME PURE ET LE BENCHMARK
# =====================================================================
print("\n========== SAUVEGARDE DES PRÉDICTIONS ET DES MÉTRIQUES ==========\n")

# 1) Injection des fréquences prédites (lambda_hat_NB) dans les bases
df_train["Pred_Freq_NB"] = model_nb.predict(df_train)
df_val["Pred_Freq_NB"]   = model_nb.predict(df_val)

# Mise à jour des fichiers pickles enrichis
save_pkl(df_train, "df_train")
save_pkl(df_val, "df_val")
print("[OK] Fréquences prédites (Pred_Freq_NB) ajoutées et sauvegardées dans df_train, df_val")

# 2) Sauvegarde de l'objet modèle ajusté
save_pkl(model_nb, "model_nb_glm")
print("[OK] Modèle GLM Binomiale Négative sauvegardé ('model_nb_glm.pkl').")

# 3) Extraction et structuration du tableau de métriques globales
metrics_nb = pd.DataFrame([{
    "Modele": "Binomiale Négative (NB2)",
    "Famille": "Fréquence",
    "Log_Likelihood": float(model_nb.llf),
    "AIC": float(model_nb.aic),
    "BIC": float(getattr(model_nb, 'bic_llf', model_nb.bic)),
    "Deviance": float(model_nb.deviance),
    "Pearson_Chi2": float(model_nb.pearson_chi2),
    "Ratio_Dispersion": float(model_nb.pearson_chi2 / model_nb.df_resid)
}])

save_tab(metrics_nb, "metrics_frequence_nb")
save_pkl(metrics_nb, "metrics_frequence_nb")
print("[OK] Tableau des métriques de fréquence sauvegardé dans output/.\n")

# =====================================================================
# check_freq_dist.py -- estimation sur le jeu Test pour GLM 
# test du Xgboost sur le jeu Test et comparaison entre les deux méthodes 
# Projet : Tarification Automobile Non-Vie (GLM vs XGBoost)
# =====================================================================


from config import *
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import load_pkl, save_pkl, save_fig

# --- 1) Fonctions de Métriques Actuarielles ---------------------------

def tweedie_deviance(y_true, y_pred, weights, p=1.5):
    """
    Calcule la déviance moyenne pondérée de Tweedie pour p = 1.5.
    """
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 1e-8)  # Évite la division par zéro
    
    # Formule de déviance unitaire Tweedie (1 < p < 2)
    dev = 2 * (
        (np.power(y_true, 2 - p) / ((1 - p) * (2 - p)))
        - (y_true * np.power(y_pred, 1 - p) / (1 - p))
        + (np.power(y_pred, 2 - p) / (2 - p))
    )
    return np.sum(weights * dev) / np.sum(weights)

def gini_coefficient(y_true, y_pred, weights):
    """
    Calcule le coefficient de Gini normalisé (basé sur la courbe de Lorenz).
    Compatible avec NumPy 2.0+ (np.trapezoid) et versions antérieures (np.trapz).
    """
    # Sélection de la bonne fonction d'intégration selon la version de NumPy
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    
    df_temp = pd.DataFrame({"true": y_true, "pred": y_pred, "weight": weights})
    
    # Tri par prédiction croissante
    df_temp = df_temp.sort_values(by="pred", ascending=True)
    
    cum_weights = np.cumsum(df_temp["weight"]) / np.sum(df_temp["weight"])
    cum_claims  = np.cumsum(df_temp["true"] * df_temp["weight"]) / np.sum(df_temp["true"] * df_temp["weight"])
    
    # Aire sous la courbe d'ordonnancement (Gini de la prédiction)
    gini_pred = 1 - 2 * trapz_func(cum_claims, cum_weights)
    
    # Tri parfait (par vraies valeurs) pour normalisation
    df_perfect = df_temp.sort_values(by="true", ascending=True)
    cum_perfect = np.cumsum(df_perfect["true"] * df_perfect["weight"]) / np.sum(df_perfect["true"] * df_perfect["weight"])
    gini_max = 1 - 2 * trapz_func(cum_perfect, cum_weights)
    
    return gini_pred / gini_max

# --- 2) Chargement des données & du modèle calibré --------------------
print("Chargement de l'échantillon TEST et du modèle calibré...")
df_test = load_pkl("df_test")
model_tweedie = load_pkl("model_tweedie_calibrated")
calib_params  = load_pkl("calibration_params_tweedie")

features = [
    "DrivAge", "VehAge", "VehPower", "BonusMalus", "Density",
    "Area", "VehGas", "VehBrand"
]

cat_cols = [col for col in ["Area", "VehGas", "VehBrand"] if col in df_test.columns]
for col in cat_cols:
    df_test[col] = df_test[col].astype("category")

# Définition de la cible et des poids sur le jeu TEST
df_test["PurePrem_Obs"] = df_test["ChargeTot"] / df_test["Exposure"]
X_test = df_test[features]
y_test = df_test["PurePrem_Obs"].values
w_test = df_test["Exposure"].values

# --- 3) Prédiction du modèle XGBoost Tweedie sur le TEST ---------------
dtest = xgb.DMatrix(X_test, enable_categorical=True)

# Prédiction avec le nombre d'arbres optimal issu de la calibration
best_iteration = calib_params.get("best_iteration", None)
if best_iteration is not None:
    df_test["Pred_PurePrem_XGB"] = model_tweedie.predict(dtest, iteration_range=(0, best_iteration + 1))
else:
    df_test["Pred_PurePrem_XGB"] = model_tweedie.predict(dtest)

# --- 4) Reconstitution de la Prime Pure GLM sur le TEST --------------
print("\nReconstitution de la Prime Pure GLM...")

# 1. Vérification / Calcul de la Fréquence GLM
if "Pred_Freq_NB" in df_test.columns:
    freq_pred = df_test["Pred_Freq_NB"]
elif "Pred_Freq_Poisson" in df_test.columns:
    freq_pred = df_test["Pred_Freq_Poisson"]
elif "Pred_Freq" in df_test.columns:
    freq_pred = df_test["Pred_Freq"]
else:
    print("  -> Colonne de fréquence absente dans df_test. Calcul direct via 'model_nb_glm.pkl'...")
    try:
        model_freq = load_pkl("model_nb_glm")
    except Exception:
        model_freq = load_pkl("model_poisson_glm")
    freq_pred = model_freq.predict(df_test)
    df_test["Pred_Freq_NB"] = freq_pred

# 2. Vérification / Calcul de la Sévérité GLM
if "Pred_Sev_LogNormal" in df_test.columns:
    sev_pred = df_test["Pred_Sev_LogNormal"]
elif "Pred_Sev" in df_test.columns:
    sev_pred = df_test["Pred_Sev"]
else:
    print("  -> Colonne de sévérité absente dans df_test. Calcul direct via 'model_lognormal_glm.pkl'...")
    model_sev = load_pkl("model_lognormal_glm")
    sigma2_hat = float(model_sev.scale)
    sev_pred = np.exp(model_sev.predict(df_test) + sigma2_hat / 2.0)
    df_test["Pred_Sev_LogNormal"] = sev_pred

# 3. Calcul de la Prime Pure GLM (Fréquence x Sévérité)
df_test["Pred_PurePrem_GLM"] = freq_pred * sev_pred

print(f"Prime Pure GLM Moyenne (Test)     : {df_test['Pred_PurePrem_GLM'].mean():.2f} €")
print(f"Prime Pure XGBoost Moyenne (Test)  : {df_test['Pred_PurePrem_XGB'].mean():.2f} €")
# --- 5) Calcul du Tableau Comparatif GLM vs XGBoost (sur TEST) -------
models_to_eval = {
    "GLM Baseline (Poisson x Log-Normale)": df_test["Pred_PurePrem_GLM"].values,
    "XGBoost Standalone (Tweedie)": df_test["Pred_PurePrem_XGB"].values
}

results = []

for name, y_pred in models_to_eval.items():
    # 1. Charge globale et Ratio d'Équilibre
    
    total_obs  = np.sum(y_test * w_test)
    total_pred = np.sum(y_pred * w_test)
    ratio_eq   = total_pred / total_obs
    
    # 2. RMSE et MAE pondérées
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))
    mae  = mean_absolute_error(y_test, y_pred, sample_weight=w_test)
    
    # 3. Déviance Tweedie (p=1.5)
    dev_tweedie = tweedie_deviance(y_test, y_pred, w_test, p=1.5)
    
    # 4. Coefficient de Gini Normalisé
    gini = gini_coefficient(y_test, y_pred, w_test)
    
    results.append({
        "Modèle": name,
        "Prime Moy. Prédite (€)": total_pred / np.sum(w_test),
        "Ratio Équilibre": ratio_eq,
        "RMSE (€)": rmse,
        "MAE (€)": mae,
        "Déviance Tweedie": dev_tweedie,
        "Indice Gini Normalisé": gini
    })

df_results = pd.DataFrame(results)

print("\n" + "="*95)
print("          ÉVALUATION COMPARED SUR L'ÉCHANTILLON TEST (HORS-DOUBLON)")
print("="*95)
print(df_results.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
print("="*95 + "\n")

# --- 6) Graphique : Comparaison des Primes Moyennes par Décite ---------
print("Génération du graphique de comparaison par décile de risque...")

df_test["Decile_XGB"] = pd.qcut(df_test["Pred_PurePrem_XGB"], q=10, labels=False, duplicates="drop") + 1

decile_analysis = df_test.groupby("Decile_XGB").apply(
    lambda g: pd.Series({
        "Observed": np.average(g["PurePrem_Obs"], weights=g["Exposure"]),
        "GLM": np.average(g["Pred_PurePrem_GLM"], weights=g["Exposure"]),
        "XGBoost": np.average(g["Pred_PurePrem_XGB"], weights=g["Exposure"])
    })
).reset_index()

fig, ax = plt.subplots(figsize=(9, 5))
x = decile_analysis["Decile_XGB"]

ax.plot(x, decile_analysis["Observed"], marker="o", color="black", label="Prime Observée", linewidth=2)
ax.plot(x, decile_analysis["GLM"], marker="s", color="#D9534F", linestyle="--", label="GLM Baseline")
ax.plot(x, decile_analysis["XGBoost"], marker="^", color="#0275D8", linestyle="-", label="XGBoost Tweedie")

ax.set_title("Tarification par décile de risque (Échantillon Test)", fontsize=12, pad=12)
ax.set_xlabel("Décile de risque prédit (XGBoost)", fontsize=10)
ax.set_ylabel("Prime Pure Moyenne (€)", fontsize=10)
ax.set_xticks(x)
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
save_fig(fig, "comparaison_déciles_test")

# --- 7) Sauvegarde Finale du jeu Test enrichi -------------------------
save_pkl(df_test, "df_test")
df_results.to_csv("output/tableau_comparatif_test.csv", index=False)

print("Évaluation sur le jeu de TEST terminée et résultats enregistrés avec succès !")
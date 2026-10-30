import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from config import load_pkl, save_pkl, save_fig

# Style graphique épuré conforme à vos visuels
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

# --- 1) Chargement strict (Train & Validation uniquement) -------------
print("Chargement des ensembles Train et Validation (Test exclu)...")
df_train = load_pkl("df_train")
df_val   = load_pkl("df_val")

features = [
    "DrivAge", "VehAge", "VehPower", "BonusMalus", "Density",
    "Area", "VehGas", "VehBrand"
]

cat_cols = [col for col in ["Area", "VehGas", "VehBrand"] if col in df_train.columns]

for df in [df_train, df_val]:
    df["PurePrem_Obs"] = df["ChargeTot"] / df["Exposure"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

X_train, y_train, w_train = df_train[features], df_train["PurePrem_Obs"], df_train["Exposure"]
X_val,   y_val,   w_val   = df_val[features],   df_val["PurePrem_Obs"],   df_val["Exposure"]

# --- 2) Construction des matrices DMatrix -----------------------------
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
dval   = xgb.DMatrix(X_val,   label=y_val,   weight=w_val,   enable_categorical=True)

# --- 3) Définition des hyperparamètres de calibration ----------------
p_tweedie = 1.5
params_tweedie = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": p_tweedie,
    "eval_metric": ["tweedie-nloglik@1.5", "rmse"],  # Suivi Tweedie + RMSE
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 10,
    "random_state": 42
}

# Dictionnaire pour enregistrer l'historique d'apprentissage à chaque itération
evals_result = {}

# --- 4) Entraînement avec suivi de la convergence ---------------------
print("\nLancement du processus de calibration...")
num_boost_round = 800

model_calib = xgb.train(
    params=params_tweedie,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, "train"), (dval, "val")],
    evals_result=evals_result,
    early_stopping_rounds=50,
    verbose_eval=100
)

best_iter = model_calib.best_iteration
best_score = model_calib.best_score
print(f"\n[CALIBRATION RÉUSSIE]")
print(f"-> Nombre d'itérations optimal (best_iteration) : {best_iter}")
print(f"-> Meilleure Déviance Tweedie (Validation)       : {best_score:.4f}")

# --- 5) Graphique 1 : Évolution de la RMSE en fonction des itérations -
rmse_val = evals_result["val"]["rmse"]
iterations = np.arange(1, len(rmse_val) + 1)

fig1, ax1 = plt.subplots(figsize=(7, 6))
ax1.plot(iterations, rmse_val, color="black", linewidth=1.2)
ax1.set_title("Évolution de la RMSE en fonction du nombre d'itérations", fontsize=12, pad=12)
ax1.set_xlabel("Nombre d'itérations", fontsize=10)
ax1.set_ylabel("RMSE", fontsize=10)
ax1.grid(True, linestyle="-", alpha=0.3, color="gray")
ax1.set_facecolor("white")
fig1.patch.set_facecolor("white")

plt.tight_layout()
save_fig(fig1, "evolution_rmse_iterations")

# --- 6) Graphique 2 : Importance des variables (Gain) -----------------
# Extraction du score d'importance (Gain)
score_dict = model_calib.get_score(importance_type="gain")
df_importance = pd.DataFrame(
    list(score_dict.items()), 
    columns=["Feature", "Importance"]
).sort_values(by="Importance", ascending=True)

# Normalisation relative pour avoir une somme à 1 (ou échelle identique à votre image)
total_imp = df_importance["Importance"].sum()
df_importance["Importance_Norm"] = df_importance["Importance"] / total_imp

fig2, ax2 = plt.subplots(figsize=(8, 6))
bars = ax2.barh(
    df_importance["Feature"], 
    df_importance["Importance_Norm"], 
    color="#C0C0C0", 
    edgecolor="none",
    height=0.6
)

ax2.set_title("Importance des variables (Gain)", fontsize=12, pad=12)
ax2.set_xlabel("Importance relative", fontsize=10)
ax2.grid(False)
ax2.set_facecolor("white")
fig2.patch.set_facecolor("white")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
save_fig(fig2, "importance_variables_tweedie")

# --- 7) Sauvegarde des paramètres de calibration ---------------------
calibration_params = {
    "best_iteration": best_iter,
    "best_score_val": best_score,
    "tweedie_variance_power": p_tweedie,
    "learning_rate": params_tweedie["learning_rate"],
    "max_depth": params_tweedie["max_depth"],
    "subsample": params_tweedie["subsample"],
    "colsample_bytree": params_tweedie["colsample_bytree"]
}

save_pkl(calibration_params, "calibration_params_tweedie")
save_pkl(model_calib, "model_tweedie_calibrated")

print("\nParamètres et graphiques de calibration enregistrés sans toucher au jeu de TEST.")
# =====================================================================
# 00_config.py  --  Configuration globale du projet
# Projet : Tarification Automobile Non-Vie (Benchmark GLM vs XGBoost)
# ---------------------------------------------------------------------
# Ce script est importé par tous les scripts de modélisation. Il :
#   - charge les bibliothèques fondamentales,
#   - détermine la racine du projet (remonte du dossier Code/ vers le parent),
#   - crée l'arborescence de sortie (data, output/figures, output/tables...),
#   - définit les paramètres métier de tarification (chargements, réassurance),
#   - fournit les fonctions utilitaires (sauvegardes, test Khi-2).
# =====================================================================

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Suppression des avertissements non critiques
warnings.filterwarnings("ignore")

# --- 1. Gestion robuste de la racine du projet -----------------------
def get_proj_root() -> Path:
    """Remonte d'un niveau depuis Code/ vers la racine du projet."""
    try:
        # Mode script (.py dans Code/)
        return Path(__file__).resolve().parent.parent
    except NameError:
        # Mode interactif
        cwd = Path.cwd()
        return cwd.parent if cwd.name == "Code" else cwd

PROJ = get_proj_root()

# Arborescence de répertoires
DIR_DATA    = PROJ / "data"
DIR_FIG     = PROJ / "output" / "figures"
DIR_TAB     = PROJ / "output" / "tables"
DIR_DERIVED = PROJ / "output" / "derived"

for d in [DIR_DATA, DIR_FIG, DIR_TAB, DIR_DERIVED]:
    d.mkdir(parents=True, exist_ok=True)

# --- 2. Paramètres Métier de Tarification ----------------------------
SEUIL_GRAVE  = 10000.0   # Seuil des sinistres "graves" (en euros)
CHARGE_FRAIS = 0.20      # tau : chargement frais de gestion / commissions (20%)
CHARGE_SECU  = 0.05      # alpha : chargement de sécurité (5%)
TAUX_TAXE    = 0.15      # Taxe fiscale sur les conventions d'assurance (15%)

# Traité de réassurance non-proportionnelle (Excess of Loss - XS)
REA_PRIORITE = 50000.0   # Rétention / Priorité de la cédante
REA_PORTEE   = 1000000.0 # Portée de la couverture réassureur

# Reproductibilité
SEED = 2026
np.random.seed(SEED)

# --- 3. Thème et Charte Graphique -----------------------------------
PALETTE = ["#1F3864", "#C00000", "#70AD47", "#ED7D31", "#7F7F7F"]
sns.set_theme(style="whitegrid", palette=sns.color_palette(PALETTE))

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "figure.titlesize": 13,
    "figure.autolayout": True
})

# --- 4. Fonctions Utilitaires ---------------------------------------

def save_fig(fig_or_plt, name: str, w: float = 8, h: float = 5) -> Path:
    """Sauvegarde standardisée d'une figure (PNG, 110 DPI)."""
    path = DIR_FIG / f"{name}.png"
    if hasattr(fig_or_plt, "savefig"):
        fig_or_plt.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    else:
        plt.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close()
    return path

def save_tab(df: pd.DataFrame, name: str) -> Path:
    """Sauvegarde d'un DataFrame au format CSV."""
    path = DIR_TAB / f"{name}.csv"
    df.to_csv(path, index=False)
    return path

def save_pkl(obj, name: str) -> Path:
    """Sauvegarde d'un objet Python (modèle, DataFrame intermédiaire)."""
    path = DIR_DERIVED / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path

def load_pkl(name: str):
    """Chargement d'un objet Pickle."""
    path = DIR_DERIVED / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def khi2_adequation(obs_counts: np.ndarray, prob_theo: np.ndarray, n_params: int) -> dict:
    """
    Test d'adéquation du Khi-deux pour les lois de fréquence/loi discrète.
    Regroupement des classes à effectif théorique < 5 par la fin.
    """
    obs = np.array(obs_counts, dtype=float)
    N = np.sum(obs)
    exp = np.array(prob_theo, dtype=float) * N

    while len(exp) > 2 and exp[-1] < 5:
        exp[-2] += exp[-1]
        obs[-2] += obs[-1]
        exp = exp[:-1]
        obs = obs[:-1]

    D = np.sum((obs - exp) ** 2 / exp)
    ddl = len(exp) - 1 - n_params

    if ddl <= 0:
        return {"D": float(D), "ddl": ddl, "pvalue": np.nan, "decision": "ddl <= 0"}

    pval = float(stats.chi2.sf(D, df=ddl))
    seuil_5pct = float(stats.chi2.ppf(0.95, df=ddl))

    return {
        "D": float(D),
        "ddl": int(ddl),
        "pvalue": pval,
        "seuil_5pct": seuil_5pct,
        "decision": "Rejet de H0" if D > seuil_5pct else "Non rejet de H0"
    }

print(f"[00_config] Racine du projet configurée : {PROJ}")
print("[00_config] Configuration tarification chargée.")
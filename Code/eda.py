# =====================================================================
# eda.py  --  Analyse exploratoire / Études univariées
# Projet : Tarification Automobile Non-Vie (GLM vs XGBoost)
# ---------------------------------------------------------------------
# Identification des variables discriminantes par croisement avec la
# variable à expliquer (fréquence / coût), analyse de l'âge conducteur,
# matrice de corrélation de Pearson.
# =====================================================================

import sys
from pathlib import Path

# Importer la configuration globale et ses utilitaires (np, pd, plt, sns, stats, save_...)
sys.path.append(str(Path(__file__).resolve().parent))
from config import *

print("\n==========  ANALYSE EXPLORATOIRE ==========\n")

# --- Chargement des objets sauvegardés à l'étape 01 ------------------
policies = load_pkl("policies_full")
claims = load_pkl("claims")

freq_moyenne_globale = policies["ClaimNb"].sum() / policies["Exposure"].sum()

# --- 1) Fréquence par facteur (variables discrétisées) ---------------
def freq_par_facteur(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Calcule l'exposition, le nombre de sinistres et la fréquence par modalité."""
    res = df.groupby(var, observed=False).agg(
        Exposition=("Exposure", "sum"),
        Sinistres=("ClaimNb", "sum")
    ).reset_index()
    res["Frequence"] = res["Sinistres"] / res["Exposition"]
    return res.sort_values(by=var)

factors_freq = [
    "DrivAgeBand", "BonusMalusBand", "VehAgeBand", 
    "VehPowerBand", "VehGas", "Area", "DensityBand"
]

for v in factors_freq:
    if v not in policies.columns:
        continue
    
    tab = freq_par_facteur(policies, v)
    save_tab(tab, f"freq_{v}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=tab, x=v, y="Frequence", color=PALETTE[0], ax=ax)
    ax.axhline(
        freq_moyenne_globale, 
        color=PALETTE[1], 
        linestyle="--", 
        label=f"Moyenne portefeuille ({freq_moyenne_globale:.4f})"
    )
    ax.set_title(f"Fréquence empirique par {v}")
    ax.set_xlabel(v)
    ax.set_ylabel("Fréquence (sinistres / année-risque)")
    ax.legend()
    save_fig(fig, f"freq_{v}")

# --- 2) Analyse spécifique : Âge du conducteur (DrivAge) --------------
# Graphique double : distribution des conducteurs + fréquence empirique par âge
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# A) Distribution de l'exposition par âge
sns.histplot(
    data=policies, 
    x="DrivAge", 
    weights="Exposure", 
    discrete=True, 
    color=PALETTE[0], 
    ax=ax1
)
ax1.set_title("Distribution de l'exposition par âge du conducteur")
ax1.set_xlabel("Âge du conducteur (DrivAge)")
ax1.set_ylabel("Exposition cumulée (années-risque)")

# B) Fréquence empirique observée par âge exact
age_freq = policies.groupby("DrivAge").agg(
    Exposition=("Exposure", "sum"),
    Sinistres=("ClaimNb", "sum")
).reset_index()
age_freq["Frequence"] = age_freq["Sinistres"] / age_freq["Exposition"]

sns.lineplot(data=age_freq, x="DrivAge", y="Frequence", color=PALETTE[1], linewidth=2, ax=ax2)
ax2.axhline(
    freq_moyenne_globale, 
    color=PALETTE[4], 
    linestyle="--", 
    label=f"Moyenne globale ({freq_moyenne_globale:.4f})"
)
ax2.set_title("Fréquence empirique par âge du conducteur")
ax2.set_xlabel("Âge du conducteur (DrivAge)")
ax2.set_ylabel("Fréquence (sinistres / an)")
ax2.legend()

save_fig(fig, "distribution_frequence_drivage")
save_tab(age_freq, "freq_drivage_exact")

# --- 3) Matrice de corrélation de Pearson (Heatmap) ------------------
# Exclusion explicite de ClaimNb et LogDensity
num_cols = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density", "Exposure"]
num_cols_exist = [c for c in num_cols if c in policies.columns]

corr_matrix = policies[num_cols_exist].corr(method="pearson")
save_tab(corr_matrix.reset_index(), "correlation_pearson")

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    vmin=-1, 
    vmax=1, 
    linewidths=0.5, 
    cbar_kws={"label": "Coefficient de Pearson"},
    ax=ax
)
ax.set_title("Heatmap des coefficients de corrélation de Pearson")
save_fig(fig, "heatmap_correlation")

# --- 4) Coût moyen par facteur (sur sinistres attritionnels) ---------
def cout_par_facteur(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Calcule l'effectif, le coût moyen et la médiane des sinistres par modalité."""
    res = df.groupby(var, observed=False).agg(
        NbSin=("ClaimAmount", "count"),
        CoutMoyen=("ClaimAmount", "mean"),
        Mediane=("ClaimAmount", "median")
    ).reset_index()
    return res.sort_values(by=var)

claims_attr = claims[claims["Grave"] == False].copy()

factors_cost = ["DrivAgeBand", "VehAgeBand", "VehGas", "Area"]

for v in factors_cost:
    if v not in claims_attr.columns:
        continue
    tab = cout_par_facteur(claims_attr, v)
    save_tab(tab, f"cout_{v}")

# --- 5) Distribution du coût des sinistres ---------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(claims["ClaimAmount"], bins=60, color=PALETTE[2], log_scale=True, ax=ax)
ax.axvline(
    SEUIL_GRAVE, 
    color=PALETTE[1], 
    linestyle="--", 
    label=f"Seuil graves ({SEUIL_GRAVE:,.0f} EUR)".replace(",", " ")
)
ax.set_title("Distribution du coût des sinistres (échelle log)")
ax.set_xlabel("Coût du sinistre (EUR, échelle log10)")
ax.set_ylabel("Effectif")
ax.legend()
save_fig(fig, "distribution_cout")

# --- 6) Distribution du nombre de sinistres par contrat --------------
tab_nb = policies["ClaimNb"].value_counts().reset_index()
tab_nb.columns = ["ClaimNb", "N"]
tab_nb = tab_nb.sort_values("ClaimNb")
tab_nb["Proportion"] = (tab_nb["N"] / tab_nb["N"].sum()).round(5)
save_tab(tab_nb, "distribution_nbsin")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=tab_nb, x="ClaimNb", y="N", color=PALETTE[0], ax=ax)
ax.set_yscale("log")
ax.set_title("Distribution du nombre de sinistres par contrat")
ax.set_xlabel("Nombre de sinistres")
ax.set_ylabel("Nombre de contrats (échelle log10)")
save_fig(fig, "distribution_nbsin")


print("\n OK - Graphiques de l'âge, Heatmap Pearson et résultats générés dans output/figures/ et output/tables/\n")
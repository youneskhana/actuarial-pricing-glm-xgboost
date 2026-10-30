# =====================================================================
# data_prep.py  --  Préparation et fiabilisation de la base
# Projet : Tarification Automobile Non-Vie (GLM vs XGBoost)
# =====================================================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

print("\n========== 01 - PRÉPARATION ET FIABILISATION DES DONNÉES ==========\n")

# --- 1) Chargement des bases -----------------------------------------
print("Chargement des données freMTPL2 (OpenML)...")
try:
    freq_raw = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    sev_raw = fetch_openml(data_id=41215, as_frame=True, parser="auto").frame
except Exception as e:
    print(f"[Info] Chargement OpenML indisponible ({e}). Tentative via CSV locaux...")
    freq_raw = pd.read_csv(DIR_DATA / "raw" / "freMTPL2freq.csv")
    sev_raw = pd.read_csv(DIR_DATA / "raw" / "freMTPL2sev.csv")

freq = freq_raw.copy()
sev = sev_raw.copy()

print(f"Base contrats  : {len(freq):,} polices, {freq.shape[1]} variables".replace(",", " "))
print(f"Base sinistres : {len(sev):,} enregistrements".replace(",", " "))

# --- 2) Fiabilisation contrats ---------------------------------------
n0 = len(freq)
freq = freq[freq["Exposure"] > 0].copy()
freq["Exposure"] = np.minimum(freq["Exposure"], 1.0)
freq["ClaimNb"] = np.minimum(freq["ClaimNb"], 4)
print(f"Nettoyage exposition : {n0 - len(freq):,} lignes retirées (Exposure <= 0)".replace(",", " "))

# --- 3) Agrégation et fusion sinistres -------------------------------
sev_agg = sev.groupby("IDpol").agg(
    ChargeTot=("ClaimAmount", "sum"),
    NbSinSev=("ClaimAmount", "count")
).reset_index()

freq = pd.merge(freq, sev_agg, on="IDpol", how="left")
freq["ChargeTot"] = freq["ChargeTot"].fillna(0.0)
freq["NbSinSev"] = freq["NbSinSev"].fillna(0)
freq["ClaimAmount"] = freq["ChargeTot"]

# Harmonisation : Pour la sévérité, utiliser le nombre réel de sinistres payés si disponible
freq["ClaimNb_Sev"] = np.where(freq["NbSinSev"] > 0, freq["NbSinSev"], freq["ClaimNb"])

# --- 4) Création des variables tarifaires (Avant le Split) -----------
freq["LogDensity"] = np.log(freq["Density"])

freq["DrivAgeBand"] = pd.cut(
    freq["DrivAge"], 
    bins=[17, 22, 26, 42, 74, 110], 
    labels=["18-22", "23-26", "27-42", "43-74", "75+"]
)
freq["VehAgeBand"] = pd.cut(
    freq["VehAge"], 
    bins=[-1, 0, 5, 12, 110], 
    labels=["0", "1-5", "6-12", "13+"]
)
freq["VehPowerBand"] = pd.cut(
    freq["VehPower"], 
    bins=[0, 5, 7, 9, 110], 
    labels=["<=5", "6-7", "8-9", "10+"]
)
freq["BonusMalusBand"] = pd.cut(
    freq["BonusMalus"], 
    bins=[0, 50, 60, 90, 350], 
    labels=["50", "51-60", "61-90", "91+"]
)
freq["DensityBand"] = pd.qcut(freq["Density"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

# Sécurisation des catégories en type chaîne
cat_cols = ['Area', 'VehGas', 'VehBrand', 'Region', 'DrivAgeBand', 'VehAgeBand', 'VehPowerBand', 'BonusMalusBand', 'DensityBand']
for col in cat_cols:
    if col in freq.columns:
        freq[col] = freq[col].astype(str)

# --- 5) Base sinistres enrichie --------------------------------------
claims = pd.merge(
    sev[sev["ClaimAmount"] > 0],
    freq[['IDpol', 'DrivAgeBand', 'VehAgeBand', 'VehPowerBand', 'BonusMalusBand', 'DensityBand', 'LogDensity', 'VehGas', 'Area', 'VehBrand', 'Region']],
    on="IDpol",
    how="inner"
)
claims["Grave"] = claims["ClaimAmount"] >= SEUIL_GRAVE

print(f"Sinistres rattachés à une police : {len(claims):,}".replace(",", " "))
print(f"dont graves (>= {SEUIL_GRAVE:,.0f} EUR) : {claims['Grave'].sum():,} ({100 * claims['Grave'].mean():.2f}%)".replace(",", " "))

# --- 6) Découpage Stratifié (70% Train / 20% Val / 10% Test) --------
df_train, df_temp = train_test_split(
    freq, test_size=0.30, random_state=SEED, stratify=freq["ClaimNb"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=10/30, random_state=SEED, stratify=df_temp["ClaimNb"]
)

print(f"\nSplit stratifié (70/20/10) effectué :")
print(f"  - Train      : {len(df_train):,} polices ({len(df_train)/len(freq):.0%})".replace(",", " "))
print(f"  - Validation : {len(df_val):,} polices ({len(df_val)/len(freq):.0%})".replace(",", " "))
print(f"  - Test       : {len(df_test):,} polices ({len(df_test)/len(freq):.0%})".replace(",", " "))

print(f"\nFréquence empirique Train      : {df_train['ClaimNb'].sum() / df_train['Exposure'].sum():.4f}")
print(f"Fréquence empirique Validation : {df_val['ClaimNb'].sum() / df_val['Exposure'].sum():.4f}")
print(f"Fréquence empirique Test       : {df_test['ClaimNb'].sum() / df_test['Exposure'].sum():.4f}")

# --- 7) Indicateurs portefeuille -------------------------------------
expo_tot = freq['Exposure'].sum()
nb_sin = freq['ClaimNb'].sum()
charge_tot = freq['ClaimAmount'].sum()

indicateur_portef = pd.DataFrame({
    "Indicateur": [
        "Exposition (années-risque)", "Nombre de sinistres", "Charge totale (EUR)",
        "Fréquence (sin/an)", "Coût moyen (EUR)", "Prime pure empirique (EUR/an)"
    ],
    "Valeur": [
        f"{expo_tot:,.1f}".replace(",", " "),
        f"{int(nb_sin):,}".replace(",", " "),
        f"{charge_tot:,.2f}".replace(",", " "),
        f"{nb_sin / expo_tot:.4f}",
        f"{charge_tot / nb_sin:,.2f}".replace(",", " ") if nb_sin > 0 else "0",
        f"{charge_tot / expo_tot:,.2f}".replace(",", " ")
    ]
})

print("\n--- INDICATEURS DU PORTEFEUILLE ---")
print(indicateur_portef.to_string(index=False))

# --- 8) Sauvegardes --------------------------------------------------
save_pkl(freq, "policies_full")
save_pkl(df_train, "df_train")
save_pkl(df_val, "df_val")
save_pkl(df_test, "df_test")
save_pkl(claims, "claims")
save_tab(indicateur_portef, "01_indicateur_portefeuille")

print("\n OK - Bases enregistrées (df_train.pkl, df_val.pkl, df_test.pkl, claims.pkl)\n")
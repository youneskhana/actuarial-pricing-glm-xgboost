import subprocess
import time

# Chemins exacts basés sur vos fichiers dans le dossier Code/
scripts = [
    "Code/config.py",
    "Code/data_prep.py",
    "Code/eda.py",
    "Code/check_freq_dist.py",
    "Code/check_sev_dist.py",
    "Code/xgb.py",
    "Code/GLM_vs_xgb.py"
]

print("==================================================")
print("   LANCEMENT DU PIPELINE ACTUARIEL COMPLET        ")
print("==================================================\n")

start_time = time.time()

for script in scripts:
    print(f"▶ Execution de : {script} ...")
    result = subprocess.run(["python", script])
    
    if result.returncode != 0:
        print(f"\n❌ ERREUR : L'exécution s'est arrêtée à l'étape {script}")
        exit(1)
    print("✔ Terminé avec succès.\n")

elapsed = time.time() - start_time
print("==================================================")
print(f"✅ PIPELINE EXÉCUTÉ EN SÉCURITÉ EN {elapsed:.2f} S")
print("==================================================")
# Tarification Actuarielle Non-Vie : GLM Baseline vs. XGBoost Tweedie

Ce projet met en place un pipeline complet de modélisation de la **Prime Pure** en assurance Automobile Non-Vie, comparant l'approche classique **GLM (Fréquence x Sévérité)** à un algorithme de Machine Learning **XGBoost**.

---

## Méthodologie Actuarielle

1. **Prétraitement & Segmentation** :
   * Découpage du portefeuille en **Train / Validation / Test**.
   * Prise en compte explicite de l'exposition au risque (`Exposure`) comme pondération.

2. **Modèle Baseline GLM (Fréquence x Sévérité)** :
   * **Fréquence** : Binomiale Négative (NB2) avec lien Log.
   * **Sévérité** : Log-Normale ajustée sur les sinistres attritionnels avec écrêtage (`SEUIL_GRAVE = 10 000 €`).

3. **Modèle XGBoost** :
   * Estimation directe de la Prime Pure sur la loi Poisson-Gamma (tweedie).
   * Calibration des hyperparamètres (profondeur, sous-échantillonnage, arrêt précoce) réalisée exclusivement via la comparaison entre `Train` et `Validation`.

---

## 📊 Résultats et Performances sur l'Échantillon Test
| Modèle | Prime Moy. Prédite (€) | Ratio d'Équilibre | RMSE (€) | MAE (€) | Indice Gini Normalisé |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GLM Baseline (NB x Log-Normale)** | 132.15 € | 1.0002 | 5,057.55 € | 1,619.74 € | **0.2847** |
| **XGBoost Standalone (Tweedie)** | 138.27 € | 1.0463 | 4,840.08 € | 253.21 € | **0.3292** |

### Graphique de segmentation par déciles de risque
![Tarification par décile](output/figures/comparaison_d%C3%A9ciles_test.png)

* **Capacité de Segmentation** : XGBoost améliore l'indice de Gini de **+4.45 points** ($0.3292$ vs $0.2847$), démontrant un meilleur ordonnancement des risques.
* **Prévention de l'Anti-Sélection** : Le GLM sur-tarifiait les premiers déciles. XGBoost épouse plus fidèlement la charge réelle observée sur les déciles 1 à 8.

---


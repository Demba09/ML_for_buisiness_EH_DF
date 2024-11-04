# Optimisation Agricole : Utilisation des Pesticides

## Contexte Business
L’objectif de ce projet est de modéliser l’usage optimal des pesticides dans l’agriculture pour :
- Maximiser les rendements agricoles,
- Réduire les coûts et se conformer aux réglementations,
- Améliorer l'image de marque à travers des pratiques écologiques et responsables.

Ce projet s'inscrit dans un cas d'utilisation réel, utilisant un ensemble de données qui capture divers paramètres agricoles (pays, produit, précipitations, température, etc.).

## Dataset
Le dataset contient 28,242 enregistrements avec les colonnes suivantes :
- **Pays (Area)** : Catégorielle
- **Produit (Item)** : Catégorielle
- **Année (Year)** : Entier
- **Rendement (hg/ha)** : Entier
- **Précipitations (mm/an)** : Float
- **Pesticides (tonnes)** : Float
- **Température (°C)** : Float

Licence du dataset : **CC0: Public Domain**

Ce dataset est utilisé pour prédire la quantité optimale de pesticides en fonction des cultures et des conditions climatiques.

## Méthodologie
Ce projet utilise des techniques de **régression** pour prédire la quantité de pesticides et de **classification** pour vérifier si cette quantité dépasse un seuil critique défini. 

## Baseline
Le modèle de base, servant de référence initiale, a obtenu les résultats suivants :
- **Score R2 (validation croisée)** : 0.0380
- **Score R2 (ensemble de test)** : 0.0328
- **RMSE (ensemble de test)** : 58607.4999

## Itération Finale
Après des itérations d’optimisation des hyperparamètres et des fonctionnalités, les performances du modèle se sont significativement améliorées :
- **Mean CV R2 Score** : 0.9999
- **Mean CV RMSE Score** : 683.5358
- **Test R2 Score** : 0.9999
- **Test RMSE** : 601.8627

### Choix et Comparaison des Modèles
Plusieurs modèles ont été testés et comparés, avec des ajustements d’hyperparamètres pour maximiser les performances et limiter l’overfitting. Une colonne `Pesticide_Class` a également été ajoutée pour classer les quantités de pesticides.
---

## Utilisation

Pour exécuter ce projet, suivez ces instructions :

### 1. Installation des dépendances

Installez les dépendances nécessaires en exécutant la commande suivante :

```bash
pip install -r requirements.txt
```

### 2. Exécution du script

Lancez le script principal avec la commande suivante :

```bash
python script.py
```

Ce script charge le dataset, entraîne un modèle, et enregistre automatiquement les résultats (métriques, courbes, modèles, etc.) dans MLflow. Vous pouvez consulter ces résultats dans l’interface MLflow pour une analyse complète des performances du modèle.

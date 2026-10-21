# Projet : Prédiction des prix immobiliers (California Housing)

# Projet : Prédiction des prix immobiliers (California Housing)

## 1. Compréhension du besoin (Le Client)
L'agence immobilière perd du temps à estimer manuellement les prix. Elle veut un outil automatique et fiable pour donner une première estimation à ses clients.

## 2. Pourquoi le Machine Learning ?
Une règle simple (ex: Prix = Surface * 3000) est trop imprécise. Un modèle de Régression (ML) capturera les tendances complexes (revenus du quartier, localisation, etc.).

## 3. La Métrique de Succès (Unique)
Le modèle doit obtenir une **MAE (Erreur Moyenne Absolue) inférieure à 0.40** (soit 40 000 $ d'erreur moyenne). Si c'est le cas, le projet est un succès.

## 4. Localisation des Données
Les données sont accessibles directement via la bibliothèque Python `scikit-learn` (dataset `fetch_california_housing`). Aucun accès à une base de données externe n'est nécessaire.

# En Maintenance:

l'application streamlit dépend d'une implémentation open source de shap désormais dépréciée pour l'interprétabilité du modèle xgboost.

Le code de l'application bidouillé est disponible dans ce dossier mais il n'est pas déployé et les fonctionnalités dépendant de shap ne sont pas disponibles (graphiques d'interprétation des prédictions sur la base de l'importance des features)

Un fork du projet est en cours de construction pour reprendre le développement du package: https://github.com/dsgibbons/shap

L'API Flask devrait fonctionné normalement pour l'instant puisqu'elle se contente de renvoyer les données des clients enregistrés dans la base de données ainsi que la prédiction de défaut de paiement par le modèle.

Elle n'est pas non plus déployé pour l'instant pour économiser un peu sur les ressources.

Le code est disponible dans le dossier Flask_api.

# Anciennes versions :

## Streamlit: repo archivé https://github.com/adnene-guessoum/Projet_scoring_credit
## Flask: repo archivé https://github.com/adnene-guessoum/scoring_credit_Flask_API

# Description du projet:

## Contexte:

Scoring de demande de crédit avec déploiement du modèle. Déterminer si un client sera en mesure de rembourser un crédit et déployer un dashboard à l'usage de l'entreprise. Le modèle est interprété avec Shap et les résultats sont présentés dans le dashboard.

## Outils:

- Python, Flask, Streamlit
- XGBoost, SHAP, BorutaShap
- données kaggle: https://www.kaggle.com/competitions/home-credit-default-risk/data
- PSQL, csv

## Repo contient:

- notebooks de développement des modèles (EDA, Feature engineering, Modèlisations et comparaisons, tests shap,...)
- notes de présentations et documents de synthèses / illustration du projet
- code de l'application streamlit
- code de l'API Flask
- CI, tests, lint, format avec Poetry, Task et github actions

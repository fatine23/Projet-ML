# Sentiment Analysis on IMDb Movie Reviews

## Introduction
L'analyse des sentiments est un cas d'utilisation de la classification de texte qui consiste à attribuer une catégorie à un texte donné. Il s'agit d'une puissante technique de traitement du langage naturel (NLP) qui permet d'analyser automatiquement ce que les gens pensent d'un sujet donné. Cela peut aider les entreprises et les particuliers à prendre rapidement des décisions plus éclairées. L'analyse des sentiments a par exemple des applications dans les médias sociaux, le service client et les études de marché.

L'objectif de ce projet est de construire un modèle capable de prédire avec précision si une critique de film est positive ou non. 

## Objectifs du projet
Exploration des données : comprenez l'ensemble de données IMDb Movie Reviews, y compris sa taille, sa structure et la répartition des sentiments.

Prétraitement des données : nettoyez et préparez les données textuelles pour la modélisation. Cela implique des tâches telles que le nettoyage du texte, la tokenisation et la gestion des déséquilibres dans les classes de sentiments.

Sélection du modèle : choisissez et mettez en œuvre un modèle d'apprentissage en profondeur approprié pour l'analyse des sentiments. Dans ce cas, une combinaison de modèles Transformer basés sur BERT est utilisée.

Formation et évaluation : entraînez le modèle choisi sur l'ensemble de données IMDb, évaluez ses performances et affinez si nécessaire. Des mesures telles que l’exactitude, la perte et éventuellement d’autres comme la précision et le rappel peuvent être prises en compte.

Visualisation des courbes d'entraînement : surveillez la progression de l'entraînement en visualisant les courbes de perte et de précision au fil des époques.

Tests et prédiction : évaluez les performances du modèle sur un ensemble de tests distinct et utilisez le modèle formé pour prédire les sentiments sur les nouveaux avis invisibles.

## Technologies utilisées
PyTorch : cadre d'apprentissage en profondeur pour la création et la formation de modèles de réseaux neuronaux.

Bibliothèque Transformers : la bibliothèque de Hugging Face pour les modèles de langage pré-entraînés comme BERT.

Visualisation des données : Matplotlib pour créer des visualisations de courbes d'entraînement.

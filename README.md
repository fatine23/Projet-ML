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

## Etapes d'études 
1/ Le choix de modèles Transformer basés sur BERT: 

-> Compréhension contextuelle : BERT (Bidirectionnel Encoder Representations from Transformers) est un modèle basé sur un transformateur qui capture les informations contextuelles du contexte gauche et droit de chaque mot dans une phrase. Cette compréhension contextuelle est cruciale pour l’analyse des sentiments, où la signification d’un mot ou d’une expression peut fortement dépendre du contexte qui l’entoure.

-> Représentations pré-entraînées : BERT est pré-entraîné sur de grandes quantités de données textuelles, ce qui lui permet d'apprendre des représentations riches et généralisables du langage. Cette pré-formation aide à capturer des modèles nuancés et une sémantique pertinents pour les tâches d'analyse des sentiments.

-> Apprentissage par transfert : les transformateurs pré-entraînés comme BERT peuvent être affinés sur des tâches spécifiques en aval, telles que l'analyse des sentiments. Cette approche d'apprentissage par transfert exploite les connaissances acquises lors de la pré-formation et les adapte à la tâche cible, nécessitant souvent moins de données étiquetées pour un réglage précis.

2/ Hyperparamètres:

Définire des hyperparamètres tels que MAX_LEN (longueur maximale de la séquence), BATCH_SIZE, NUM_CLASSES (nombre de classes de sentiments), LEARNING_RATE, NUM_EPOCHS et le point de contrôle du modèle BERT (BERT_CHECKPOINT : est une variable qui spécifie le nom ou l'identifiant du modèle BERT (Bidirectionnel Encoder Representations from Transformers) pré-entraîné de la bibliothèque Hugging Face Transformers ).

3/ Chargement et prétraitement des données:

-> Charger l'ensemble de données des critiques (reviews) de films IMDb à partir d'un fichier CSV.
-> Transformer les étiquettes de sentiment en nombres entiers (0 pour « négatif », 1 pour « positif »).
-> Diviser l'ensemble de données en ensembles de train, de validation et de test.

4/ Nettoyage du texte:

Implémente une fonction (clean_text) pour supprimer les espaces supplémentaires et les balises HTML des révisions de texte.

          def clean_text(text):
           """Removes extra whitespaces and html tags from text."""
           # remove weird spaces
           text =  " ".join(text.split())
           # remove html tags
           text = re.sub(r'<.*?>', '', text)
           return text

5/ Custom Dataset Class:

-> Définire un Dataset class PyTorch personnalisée (CustomDataset) pour gérer les données des critiques de films.
-> Tokenise et encode les phrases d'entrée à l'aide du tokenizer BERT.

       class CustomDataset(Dataset):
       def __init__(self, review, target, tokenizer, max_len, clean_text=None):
        self.clean_text = clean_text
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        y = torch.tensor(self.target[idx], dtype=torch.long)
        X = str(self.review[idx])
        if self.clean_text:
            X = self.clean_text(X)
        
        encoded_X = self.tokenizer(
            X, 
            return_tensors = 'pt', 
            max_length = self.max_len, 
            truncation=True,
            padding = 'max_length'
            )

        return {'input_ids': encoded_X['input_ids'].squeeze(),
                'attention_mask': encoded_X['attention_mask'].squeeze(),
                'labels': y}

6/Training Loop:

Implémente des fonctions (train_epoch et eval_epoch) pour former et évaluer le modèle pour une époque.
Utilise le modèle BERT pour la classification des séquences (BertForSequenceClassification).
Optimise le modèle à l'aide de l'optimiseur AdamW et d'un planificateur de taux d'apprentissage linéaire.

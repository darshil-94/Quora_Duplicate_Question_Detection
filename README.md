# Quora Duplicate Question Detection

A natural language processing (NLP) project that identifies whether two Quora questions are duplicates using advanced text similarity techniques and deep learning models.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Credits](#credits)

---

## Project Overview

This project addresses the problem of detecting semantically similar questions on Quora. Using machine learning and deep learning approaches, the model determines whether a pair of questions are duplicates, helping to reduce redundancy and improve user experience on Q&A platforms.

Key objectives:
- Preprocess and vectorize text data effectively
- Explore traditional ML and deep learning models
- Evaluate and compare performance
- Deploy an interactive Streamlit app for testing question pairs

---

## Features

- **Text preprocessing**: Tokenization, lemmatization, stopword removal
- **Feature engineering**: Word overlap, fuzzy matching, and semantic similarity scores
- **Model support**: Logistic Regression, Random Forest, XGBoost, Siamese LSTM (Deep Learning)
- **Streamlit UI**: Upload your question pair and get instant prediction
- **Visualization**: Display of similarity metrics and prediction confidence

---

## Dataset

- **Source**: [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)
- **Format**: CSV
- **Fields**: `question1`, `question2`, `is_duplicate`

> **Note**: Dataset must be placed in `data/` directory.

---

## Methodology

1. **Data Preprocessing**
   - Clean text (lowercase, remove punctuation, stopwords)
   - Convert to vector representations (TF-IDF, Word2Vec, or BERT embeddings)

2. **Feature Engineering**
   - Basic NLP features: common words, token count difference
   - Fuzzy match ratios (token sort, partial ratio)
   - Embedding distance-based similarity

3. **Modeling**
   - Train classical ML models (randomforest,xgboost) and evaluate

4. **Evaluation**
   - Accuracy, F1-score, Precision, Recall
   - Confusion matrix and ROC curve

5. **Deployment**
   - Use Streamlit to build an intuitive interface (`app.py`)
   - Load trained model using `pickel` 

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/quora-duplicate-question-detection.git
   cd quora-duplicate-question-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Preprocess and train model:**
   ```bash
   python scripts/train.py --data data/quora.csv --model output/model.pkl
   ```
2. **Run the app:**
   ```bash
   streamlit run app.py
   ```
3. **Test it:**
   - Enter two questions in the UI
   - Click "Predict" to check if they are duplicates

---


## Results

- **Best model**: Randomforestclassifier F1-score of 0.84
- **Baseline**: xgboost F1-score ~0.75

---

## Future Work

- Incorporate transformers like BERT for better semantic understanding
- Add attention mechanism to deep models
- Train on extended datasets from similar platforms (e.g., Stack Overflow)
- API deployment for third-party integration

---

## Credits

- Project developed by [Darshil Patel] (Darshil Patel)
- Dataset from Kaggle
- Libraries used: scikit-learn, matplotlib, seaborn, NLTK, Streamlit 

---

Feel free to fork, improve, and contribute!


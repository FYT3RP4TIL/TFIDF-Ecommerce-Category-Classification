# 🛍️ E-commerce Product Categorization using TFIDF

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Text Preprocessing](#text-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Best Performing Model](#best-performing-model)
- [Usage](#usage)
- [Dependencies](#dependencies)

## 🔍 Project Overview

This project aims to classify e-commerce product descriptions into four categories: Household, Electronics, Clothing & Accessories, and Books. We use Natural Language Processing (NLP) techniques and various Machine Learning algorithms to achieve this classification.

## 📊 Dataset

The dataset used in this project is from Kaggle: [E-commerce Text Classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)

It consists of two columns:
- Product description
- Label (category)

Distribution of categories:
```
Household                 6000
Electronics               6000
Clothing & Accessories    6000
Books                     6000
```

## 🛠️ Methodology

### Text Preprocessing

We use the spaCy library for text preprocessing:

1. Remove stop words
2. Lemmatize the text

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    
    return " ".join(filtered_tokens)
```

### Feature Extraction

We use TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction:

- TF (Term Frequency): Ratio of word occurrence to total words in a document
- IDF (Inverse Document Frequency): Log of ratio of total documents to documents containing the word
- TF-IDF = TF * IDF

### Machine Learning Models

We implemented and compared three different models:

1. 🔍 K-Nearest Neighbors (KNN)
2. 📊 Multinomial Naive Bayes
3. 🌳 Random Forest

All models were implemented using scikit-learn's Pipeline module for easy preprocessing and model creation.

## 📈 Results

### KNN Model

```
              precision    recall  f1-score   support

           0       0.95      0.96      0.95      1200
           1       0.97      0.95      0.96      1200
           2       0.97      0.97      0.97      1200
           3       0.97      0.98      0.97      1200

    accuracy                           0.96      4800
   macro avg       0.96      0.96      0.96      4800
weighted avg       0.96      0.96      0.96      4800
```

### Multinomial Naive Bayes Model

```
              precision    recall  f1-score   support

           0       0.92      0.96      0.94      1200
           1       0.98      0.92      0.95      1200
           2       0.97      0.97      0.97      1200
           3       0.97      0.99      0.98      1200

    accuracy                           0.96      4800
   macro avg       0.96      0.96      0.96      4800
weighted avg       0.96      0.96      0.96      4800
```

### Random Forest Model

```
              precision    recall  f1-score   support

           0       0.96      0.96      0.96      1200
           1       0.98      0.98      0.98      1200
           2       0.98      0.97      0.97      1200
           3       0.98      0.99      0.98      1200

    accuracy                           0.97      4800
   macro avg       0.97      0.97      0.97      4800
weighted avg       0.97      0.97      0.97      4800
```

## 🏆 Best Performing Model

The Random Forest model performed the best, achieving an overall accuracy of 97% and consistently high precision, recall, and F1-scores across all categories.

## 🚀 Usage

1. Prepare your data in a similar format to the Kaggle dataset.
2. Run the preprocessing function on your text data.
3. Use the scikit-learn Pipeline to vectorize the text and apply the chosen model.
4. Predict categories for new product descriptions.

## 📚 Dependencies

- Python 3.x
- scikit-learn
- spaCy
- pandas
- numpy

Install dependencies using:
```
pip install scikit-learn spacy pandas numpy
python -m spacy download en_core_web_sm
```

---

🔗 [Dataset Source](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification) | 👨‍💻 [Your Name/Organization]

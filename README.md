# SMS Spam Detection

## Project Overview

This project aims to detect spam SMS messages using various machine learning models. The project workflow includes data preprocessing, feature extraction, model development, and evaluation.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Feature Extraction](#feature-extraction)
5. [Modeling](#modeling)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [Contact](#contact)

## Introduction

SMS spam detection is crucial for filtering unwanted messages and ensuring secure communication. This project leverages machine learning techniques to classify SMS messages as spam or ham.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib, Seaborn**: Data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **Imbalanced-learn**: Handling imbalanced datasets
- **TensorFlow, Keras**: Building neural network models
- **Pickle**: Model serialization

## Data Preprocessing

1. **Data Loading**:
   - Loaded the dataset using `pd.read_csv()`.

2. **Data Cleaning**:
   - Checked for and handled missing values.
   - Converted categorical labels to numerical values (spam: 1, ham: 0).

3. **Data Splitting**:
   - Split the data into training and testing sets using `train_test_split()`.

## Feature Extraction

1. **TF-IDF Vectorizer**:
   - Converted text data into numerical vectors using `TfidfVectorizer()`.

2. **Over Sampling**:
   - Balanced the dataset using SMOTE to handle imbalanced classes.

## Modeling

1. **Neural Network**:
   - Built a simple neural network model using TensorFlow and Keras.

2. **AdaBoost Classifier**:
   - Built an AdaBoost model with a Decision Tree base estimator and evaluated its performance.

## Results

- **Neural Network**:
  - Training Accuracy: ...
  - Test Accuracy: ...

- **AdaBoost Classifier**:
  - Training Accuracy: 0.9992307692307693
  - Test Accuracy: 0.9692307692307693

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/sms-spam-detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd sms-spam-detection
   ```

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained models to predict whether new SMS messages are spam or ham.

## Conclusion

This project demonstrates the use of various machine learning models to detect SMS spam. The models were evaluated and tuned to achieve high accuracy, providing valuable insights into spam detection techniques.

## Contact

For questions or collaborations, please reach out via:

- **Email**: [osamaoabobakr112@gmail.com](mailto:osamaoabobakr112@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import tensorflow.keras as k
import pickle

# Load data
data = pd.read_csv(r"D:\Courses language programming\5_Machine Learning\Dataset For Machine Learning\Spam_Mail\mail_data.csv")

# Data cleaning
data.loc[data["Category"] == "spam", "Category"] = 1
data.loc[data["Category"] == "ham", "Category"] = 0

# Split data
x_input = data["Message"]
y_output = data["Category"]
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, train_size=0.7, random_state=42)

# Feature extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
new_x_train = feature_extraction.fit_transform(x_train)
new_x_test = feature_extraction.transform(x_test)

y_train = y_train.astype("int")
y_test = y_test.astype("int")

pickle.dump(feature_extraction, open(r"D:\Pycharm\model_pickle\Ai-Project_feature_extraction.pkl", "wb"))

# Over sampling
new_x, new_y = SMOTE().fit_resample(new_x_train, y_train)
new_x2, new_y2 = SMOTE().fit_resample(new_x_test, y_test)

# Neural network model (commented out)
# model = k.models.Sequential([
#     k.layers.Dense(128, activation="relu"),
#     k.layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="adam", loss=k.losses.CategoricalCrossentropy(), metrics=["accuracy"])
# model.fit(new_x_train, y_train, validation_data=(new_x_test, y_test))

# AdaBoost model
Adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
Adaboost.fit(new_x, new_y)

pickle.dump(Adaboost, open(r"D:\Pycharm\model_pickle\Ai-Project.bin", "wb"))

# Predict system
new_text = pd.DataFrame(data=[input()], columns=["Message"])
text = feature_extraction.transform(new_text)
prediction = Adaboost.predict(text)

if prediction[-1] == 0:
    print("Ham")
else:
    print("Spam")

print(new_text["Message"])
print("\n")
print(prediction)
```

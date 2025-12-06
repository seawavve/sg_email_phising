# EDA Insight 활용 모델
# - 결측치 처리 O
# - 데이터 불균형 해소 -> SMOTE
# - 날짜 칼럼 제외, url 칼럼 포함
# 기본 모델
import time
from imblearn.over_sampling import SMOTE

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from joblib import dump


start_total = time.time()
df = pd.read_csv("../data/CEAS_08.csv")


# Preprocessing the body text
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Transform the body text to TF-IDF features
X_tfidf = tfidf_vectorizer.fit_transform(df["body"])

# Combining other features with TF-IDF features
features = pd.DataFrame(
    X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
)
# features['date'] = df['date']
features["urls"] = df["urls"]
features["label"] = df["label"]

# Ensure there are no missing values in the features
features = features.dropna()

# Splitting the dataset into features and target
X = features.drop("label", axis=1)
# breakpoint()
y = features["label"]
# print(X.columns)


# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Building a Logistic Regression model
model_eda = LogisticRegression(max_iter=10)

model_eda.fit(X_train, y_train)
end_total = time.time()

print(f"\nModel Train execution time: {end_total - start_total:.2f} seconds")
breakpoint()

y_pred = model_eda.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report_eda = classification_report(
    y_test, y_pred, zero_division=1, output_dict=True
)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report_eda)


from fig_utils import save_confusion_matrix

fig_path = "../fig/conf_matrix_logistic_eda.png"
save_confusion_matrix(conf_matrix, fig_path)

models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "logistic_eda_model.joblib")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")

dump(model_eda, model_path)
dump(tfidf_vectorizer, vectorizer_path)

print(f"Saved model to: {model_path}")
print(f"Saved vectorizer to: {vectorizer_path}")

# 기본 모델
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from fig_utils import save_confusion_matrix

start_total = time.time()
df = pd.read_csv("../data/CEAS_08.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df["body"])

features = pd.DataFrame(
    X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
)
if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
    features["year"] = df["date"].dt.year
    features["month"] = df["date"].dt.month
    features["day"] = df["date"].dt.day
features["urls"] = df["urls"]
features["label"] = df["label"]
features = features.dropna()

X = features.drop("label", axis=1)
y = features["label"]
# print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=10, class_weight="balanced")
model.fit(X_train, y_train)

end_total = time.time()
print(f"\nModel Train execution time: {end_total - start_total:.2f} seconds")

y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

from fig_utils import save_confusion_matrix

fig_path = "../fig/conf_matrix_logistic.png"
save_confusion_matrix(conf_matrix, fig_path)

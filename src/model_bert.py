# (Model Ver.2.1) fine-tuning된 BERT 오픈모델
# Distill BERT 모델에 원래 label이 4개인데 우리 Task에 맞게 2개로 분류하여 활용함

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


# -----------------------------------
# 0. 데이터 준비 (df: body, label 컬럼 가정)
# -----------------------------------
df = pd.read_csv("../data/CEAS_08.csv")

X = df["body"].astype(str).values  # 이메일 본문
y = df["label"].values  # 0/1 라벨 (정상/피싱)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# 1. HF 모델 / 토크나이저 (이미 선언되어 있다고 가정)
# -----------------------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "cybersectony/phishing-email-detection-distilbert_v2.4.1"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "cybersectony/phishing-email-detection-distilbert_v2.4.1"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict_email(email_text):
    # Preprocess and tokenize
    inputs = tokenizer(
        email_text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get probabilities for each class
    probs = predictions[0].tolist()

    # Create labels dictionary
    labels = {
        "legitimate_email": probs[0],
        "phishing_url": probs[1],
        "legitimate_url": probs[2],
        "phishing_url_alt": probs[3],
    }

    # Determine the most likely classification
    max_label = max(labels.items(), key=lambda x: x[1])

    return {
        "prediction": max_label[0],
        "confidence": max_label[1],
        "all_probabilities": labels,
    }


# -----------------------------------
# 2. 모델 예측 → y_pred (binary: 0=정상, 1=피싱)
# -----------------------------------
def label_to_binary(label_name: str) -> int:
    # phishing 관련 라벨이면 1, 나머지는 0
    if "phishing" in label_name:
        return 1
    else:
        return 0


num_sample = 1000
y_true = y_test
y_pred = []
X_test, y_true = X_test[:num_sample], y_true[:num_sample]

for text in tqdm(X_test):
    result = predict_email(text)
    bin_label = label_to_binary(result["prediction"])
    y_pred.append(bin_label)

y_pred = np.array(y_pred)

# -----------------------------------
# 3. 평가 지표 계산 (class_report, confusion matrix)
# -----------------------------------
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, zero_division=1)

print(f"\nAccuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


from fig_utils import save_confusion_matrix

fig_path = "../fig/conf_matrix_bert.png"
save_confusion_matrix(conf_matrix, fig_path)

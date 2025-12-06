import pandas as pd
from joblib import load

# 모델 & 벡터라이저 경로
model_path = "../models/logistic_eda_model.joblib"
vectorizer_path = "../models/tfidf_vectorizer.joblib"

# 저장된 모델과 벡터라이저 불러오기
model = load(model_path)
vectorizer = load(vectorizer_path)

# -----------------------------
# 1. 새로운 이메일 데이터 불러오기
# -----------------------------
# 예시: inference_data.csv
# 컬럼: ["body", "urls"]
df_inf = pd.read_csv("../data/CEAS_08.csv")

# -----------------------------
# 2. 전처리: TF-IDF 변환
# -----------------------------
# df_inf: inference용 원본 데이터 (body, urls, label 있을 수도 있음)
# TF-IDF + urls 생성은 그대로
X_tfidf = vectorizer.transform(df_inf["body"].astype(str))
df_features = pd.DataFrame(
    X_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
)
df_features["urls"] = df_inf["urls"]

# 혹시 label이 들어왔으면 드롭
if "label" in df_features.columns:
    df_features = df_features.drop(columns=["label"])

# 모델이 학습할 때 사용한 feature 순서/이름에 정확히 맞추기
if hasattr(model, "feature_names_in_"):
    df_features = df_features[model.feature_names_in_]

predictions = model.predict(df_features)
pred_proba = model.predict_proba(df_features)[:, 1]

# -----------------------------
# 4. 결과를 원본 DF와 합치기
# -----------------------------
df_inf["pred_label"] = predictions
df_inf["pred_proba_phishing"] = pred_proba

print(df_inf.head())

# -----------------------------
# 5. CSV 파일로 저장
# -----------------------------
output_path = "../results/inference_results.csv"
df_inf.to_csv(output_path, index=False)

print(f"Inference results saved to: {output_path}")

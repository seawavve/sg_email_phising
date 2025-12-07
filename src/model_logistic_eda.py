"""
================================================================================
EDA Insight 기반 Logistic Regression 모델 - 피싱 이메일 탐지
================================================================================

이 스크립트는 비정형 텍스트 데이터(이메일 본문)를 정형 데이터로 변환하여
Logistic Regression 모델로 피싱 이메일을 탐지합니다.

주요 특징:
  - 데이터 불균형 해소: SMOTE를 사용하여 소수 클래스 오버샘플링
  - 날짜 컬럼 제외, URL 컬럼 포함
  - 비정형 -> 정형 변환: TF-IDF를 사용하여 텍스트를 수치형 벡터로 변환

작동 원리:
  1. 비정형 텍스트(이메일 본문) → TF-IDF 벡터화 → 5,000개의 정형 feature
  2. URL 존재 여부(1/0) 추가
  3. feature로 구성된 정형 데이터셋 생성
  4. 데이터 불균형 해소 (SMOTE)
  5. Logistic Regression 모델 학습 및 평가
================================================================================
"""

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


# ============================================================================
# 1. 데이터 로드 및 실행 시간 측정 시작
# ============================================================================
start_total = time.time()
df = pd.read_csv("../data/CEAS_08.csv")

print(f"전체 데이터셋 크기: {len(df)}개")


# ============================================================================
# 2. 비정형 텍스트 데이터를 정형 데이터로 변환 (TF-IDF 벡터화)
# ============================================================================
"""
TF-IDF (Term Frequency-Inverse Document Frequency)란?
  - 텍스트 데이터에서 단어의 중요도를 수치화하는 방법
  - 자주 등장하지만 모든 문서에 나타나지 않는 단어에 높은 가중치 부여
  
변환 과정:
  - 입력: 비정형 텍스트 문자열 (예: "Dear User, Click here...")
  - 출력: 정형 수치 벡터 (예: [0.0, 0.123, 0.456, ..., 0.789]) - 5,000차원
  
설정:
  - stop_words="english": 영어 불용어 제거 (the, a, is 등)
  - max_features=5000: 가장 중요한 5,000개의 단어만 선택
"""

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# 이메일 본문(body)을 TF-IDF 벡터로 변환
# 결과: 각 이메일이 5,000차원의 수치 벡터로 변환됨
X_tfidf = tfidf_vectorizer.fit_transform(df["body"])

print(f"TF-IDF 변환 완료: {X_tfidf.shape[0]}개 샘플 × {X_tfidf.shape[1]}개 feature")


# ============================================================================
# 3. TF-IDF 벡터를 DataFrame으로 변환 및 추가 feature 결합
# ============================================================================
# 희소 행렬(sparse matrix)을 밀집 행렬(dense array)로 변환
# 각 컬럼은 하나의 단어를 나타내며, 값은 해당 단어의 TF-IDF 점수
features = pd.DataFrame(
    X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
)
print(f"[TF-IDF만] Feature shape: {features.shape}")
print(f"  → 컬럼 목록 (처음 10개): {list(features.columns[:10])}")
print(f"  → 전체 컬럼 개수: {len(features.columns)}개")
print(f"  → 'urls' 컬럼 존재 여부: {'urls' in features.columns}")
print(f"  → 'label' 컬럼 존재 여부: {'label' in features.columns}")

# TF-IDF 컬럼 이름 중에 'urls'나 'label'이 포함되어 있는지 확인
urls_in_tfidf = [col for col in features.columns if 'url' in col.lower() or col == 'urls']
label_in_tfidf = [col for col in features.columns if 'label' in col.lower() or col == 'label']
if urls_in_tfidf:
    print(f"  ⚠️  주의: TF-IDF 컬럼 중에 'url'이 포함된 컬럼 발견: {urls_in_tfidf}")
if label_in_tfidf:
    print(f"  ⚠️  주의: TF-IDF 컬럼 중에 'label'이 포함된 컬럼 발견: {label_in_tfidf}")

# URL 존재 여부 feature 추가 (1: URL 있음, 0: URL 없음)
# 이는 이메일이 피싱인지 판단하는 데 중요한 신호
print(f"\n[원본 df의 urls 확인]")
print(f"  → 원본 df의 urls 샘플 (처음 5개): {df['urls'].head().tolist()}")
print(f"  → 원본 df의 urls 통계: min={df['urls'].min()}, max={df['urls'].max()}, 평균={df['urls'].mean():.2f}")

features["urls"] = df["urls"]
print(f"\n[urls 추가 후] Feature shape: {features.shape}")
print(f"  → 컬럼 개수 변화: {len(features.columns)}개 (예상: 5001개)")
print(f"  → 'urls' 컬럼 존재 여부: {'urls' in features.columns}")
if "urls" in features.columns:
    print(f"  → urls 컬럼 샘플 (처음 5개):")
    print(f"     {features['urls'].head().tolist()}")
    print(f"  → urls 컬럼 통계: min={features['urls'].min()}, max={features['urls'].max()}, 평균={features['urls'].mean():.2f}")
    print(f"  → 원본과 동일한지: {features['urls'].equals(df['urls'])}")

# 라벨 추가 (0: 정상 이메일, 1: 피싱 이메일)
print(f"\n[원본 df의 label 확인]")
print(f"  → 원본 df의 label 샘플 (처음 5개): {df['label'].head().tolist()}")
print(f"  → 원본 df의 label 통계: min={df['label'].min()}, max={df['label'].max()}, 평균={df['label'].mean():.2f}")

features["label"] = df["label"]
print(f"\n[label 추가 후] Feature shape: {features.shape}")
print(f"  → 컬럼 개수 변화: {len(features.columns)}개 (예상: 5002개)")
print(f"  → 'urls' 컬럼 존재 여부: {'urls' in features.columns}")
print(f"  → 'label' 컬럼 존재 여부: {'label' in features.columns}")
if "urls" in features.columns and "label" in features.columns:
    print(f"  → urls, label 컬럼 샘플 (처음 5개):")
    print(f"     urls: {features['urls'].head().tolist()}")
    print(f"     label: {features['label'].head().tolist()}")
    print(f"  → 전체 컬럼 수: {len(features.columns)}개 (TF-IDF {len(features.columns)-2}개 + urls 1개 + label 1개)")
    print(f"  → 원본과 비교:")
    print(f"     urls 동일 여부: {features['urls'].equals(df['urls'])}")
    print(f"     label 동일 여부: {features['label'].equals(df['label'])}")

# ============================================================================
# 4. Feature와 Label 분리
# ============================================================================
# 모델 입력(feature)과 정답(label) 분리
X = features.drop("label", axis=1)  # Feature: 5,001개 (TF-IDF 5,000개 + urls 1개)
y = features["label"]                # Label: 0(정상) 또는 1(피싱)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================================
# 5. 데이터 불균형 해소 (SMOTE)
# ============================================================================
"""
SMOTE (Synthetic Minority Oversampling Technique)란?
  - 소수 클래스(피싱 이메일)의 샘플을 합성적으로 생성하여 데이터 불균형 해소
  - 기존 데이터를 복사하는 것이 아니라, 기존 데이터들 사이에 새로운 샘플 생성
  
예시:
  - 원본: 정상 10,000개, 피싱 1,000개 → 불균형
  - SMOTE 후: 정상 10,000개, 피싱 10,000개 → 균형
"""

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print(f"SMOTE 적용 전: {len(X)}개 샘플")
print(f"SMOTE 적용 후: {len(X_res)}개 샘플")
print(f"클래스별 분포: 정상={sum(y_res==0)}개, 피싱={sum(y_res==1)}개")


# ============================================================================
# 6. Train/Test 데이터셋 분할
# ============================================================================
"""
데이터 분할 비율:
  - Train set: 90% (모델 학습용)
  - Test set: 10% (모델 평가용)
  
예시 (SMOTE 후 20,000개 샘플인 경우):
  - Train: 약 18,000개
  - Test: 약 2,000개
"""

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.1, random_state=42
)

print(f"\n데이터 분할 완료:")
print(f"  Train set: {len(X_train)}개 샘플")
print(f"  Test set: {len(X_test)}개 샘플")
print(f"  Train/Test 비율: {len(X_train)/len(X_test):.2f}:1")


# ============================================================================
# 7. Logistic Regression 모델 학습
# ============================================================================
"""
Logistic Regression이란?
  - 선형 분류 모델로, 각 feature의 가중치를 학습하여 이진 분류 수행
  - 입력: 정형 수치 벡터
  - 출력: 피싱 확률 (0.0 ~ 1.0)
  
주의: max_iter=10은 학습 반복 횟수가 적어 수렴 경고가 발생할 수 있음
      (더 나은 성능을 원하면 max_iter=100 이상 권장)
"""

model_eda = LogisticRegression(max_iter=10)

print("\n모델 학습 시작...")
model_eda.fit(X_train, y_train)
end_total = time.time()

print(f"모델 학습 완료! 소요 시간: {end_total - start_total:.2f}초")


# ============================================================================
# 8. 모델 예측 및 평가
# ============================================================================
# Test set으로 예측 수행
y_pred = model_eda.predict(X_test)

# 평가 지표 계산
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report_eda = classification_report(
    y_test, y_pred, zero_division=1, output_dict=True
)

print(f"\n{'='*60}")
print(f"모델 평가 결과")
print(f"{'='*60}")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"\n혼동 행렬 (Confusion Matrix):")
print(conf_matrix)
print(f"\n분류 리포트 (Classification Report):")
print(class_report_eda)


# ============================================================================
# 9. 혼동 행렬 시각화 및 저장
# ============================================================================
from fig_utils import save_confusion_matrix

fig_path = "../fig/conf_matrix_logistic_eda.png"
save_confusion_matrix(conf_matrix, fig_path)
print(f"\n혼동 행렬 이미지 저장: {fig_path}")


# ============================================================================
# 10. 모델 및 벡터라이저 저장
# ============================================================================
"""
저장되는 파일:
  1. logistic_eda_model.joblib: 학습된 Logistic Regression 모델
  2. tfidf_vectorizer.joblib: 학습된 TF-IDF 벡터라이저
     - 새로운 이메일을 예측할 때 동일한 방식으로 변환하기 위해 필요
"""

models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "logistic_eda_model.joblib")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")

dump(model_eda, model_path)
dump(tfidf_vectorizer, vectorizer_path)

print(f"\n{'='*60}")
print(f"모델 및 벡터라이저 저장 완료:")
print(f"  모델: {model_path}")
print(f"  벡터라이저: {vectorizer_path}")
print(f"{'='*60}")

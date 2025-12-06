# demo_logistic_eda.py
import re
import numpy as np
import pandas as pd
from joblib import load
import gradio as gr

# === 1. 저장된 모델 & 벡터라이저 로드 ===
model_path = "../models/logistic_eda_model.joblib"
vectorizer_path = "../models/tfidf_vectorizer.joblib"

model_eda = load(model_path)
tfidf_vectorizer = load(vectorizer_path)

print("모델이 기대하는 feature 수:", model_eda.n_features_in_)


# === 2. 학습 시 urls 컬럼을 근사하는 함수 ===
#    → 데모에서는 "이메일 주소 형식이 있으면 1, 없으면 0" 으로 근사
def extract_url_feature(text: str) -> float:
    """
    학습 때 features['urls']에 어떤 값이 들어갔는지 정확히는 모르지만
    데모에서는 '본문에 이메일 주소 형식이 하나라도 있으면 1, 없으면 0' 으로 근사한다.
    """
    if not isinstance(text, str):
        return 0.0

    # 간단한 이메일 패턴: something@something.something
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    has_email = re.search(email_pattern, text) is not None

    # 이메일 형식이 하나라도 있으면 1, 없으면 0
    return 1.0 if has_email else 0.0


# === 3. 예시 이메일 (드롭다운용) ===
PHISHING_EXAMPLE_1 = """Dear User,

We detected a large unauthorized withdrawal request from your wallet.

To prevent permanent loss of funds, please verify your account immediately.

Click the link below to approve or cancel the transaction:

👉 http://secure-crypto-verify.money-transfer99.pro/login

If you do not complete this verification within 1 hour, your wallet will be locked and all pending transactions will be processed automatically.

Thank you,
CryptoSafe Support

(Automated message — do not reply)"""

PHISHING_EXAMPLE_2 = """Hello,

A payment of $4,850 USD is waiting to be released to your account.

However, our system requires identity confirmation before the funds can be delivered.

To accept the payment, please complete the secure form below:

👉 http://fund-release-checker.quick-profit-exchange.co/form

This transfer will be canceled in 24 hours if the information is not submitted.

Best regards,
Fund Release Department"""

PHISHING_EXAMPLE_3 = """Dear Customer,

Your recent order from our partner marketplace is ready for shipment.

To verify your delivery address and avoid delays, please review your order details:

👉 http://global-market-access.darkdealzone.info/confirm

If you did not place this order, click the link above and cancel the transaction immediately to avoid charges.

Sincerely,
Global Shipping Control"""


NORMAL_EXAMPLE_1 = """Hi John,

This is a reminder that we have a team meeting scheduled for Thursday at 2:00 PM in Meeting Room B.

Please let me know if you need to adjust the time or if you have any agenda items you want to include.

Thanks,
HR Team"""

NORMAL_EXAMPLE_2 = """Hello,

Thank you for your purchase!

Attached is the receipt for your order placed on December 2, 2025.

If you have any questions regarding your order, feel free to contact our support team.

Best regards,
Online Store Support"""

NORMAL_EXAMPLE_3 = """Hi there,

Here are the latest updates for this week:

New machine learning courses available
Product feature improvements
Upcoming live workshops

Visit your dashboard anytime to explore more.

Have a great week!
LearnHub Team"""


# === 4. 예측 함수 ===
def predict_phishing(email_text: str):
    if not email_text or email_text.strip() == "":
        return "입력 없음", 0.0

    # 1) TF-IDF 변환
    X_tfidf = tfidf_vectorizer.transform([email_text])

    # 2) TF-IDF를 DataFrame으로 변환 (+ feature 이름 부여)
    df_features = pd.DataFrame(
        X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
    )

    # 3) urls feature 추가 (이메일 형식 존재 여부: 1 / 0)
    df_features["urls"] = extract_url_feature(email_text)

    # 4) 모델이 학습할 때 사용한 feature 순서/이름에 정확히 맞추기
    if hasattr(model_eda, "feature_names_in_"):
        df_features = df_features[model_eda.feature_names_in_]

    X_features = df_features.values

    # 5) 예측
    proba = model_eda.predict_proba(X_features)[0][1]  # 클래스 1(피싱) 확률
    label = int(proba >= 0.5)

    label_str = "피싱 (1)" if label == 1 else "정상 (0)"
    return label_str, float(proba)


# === 5. Gradio UI 정의 (드롭다운 + 자유 입력) ===
with gr.Blocks() as demo:
    gr.Markdown(
        "# 피싱 이메일 탐지 데모 (EDA Insight 기반 Logistic Regression)\n"
        "CEAS_08 데이터셋으로 학습한 Logistic Regression 모델입니다.\n\n"
        "- 전처리: 결측치 제거, SMOTE로 클래스 불균형 해소\n"
        "- 특징: TF-IDF(최대 5,000) + urls 특성 1개 (여기서는 이메일 형식 존재 여부 0/1)\n"
        "- 출력: 피싱 여부와 피싱 확률\n"
    )

    with gr.Row():
        with gr.Column():
            example_dropdown = gr.Dropdown(
                choices=[
                    ("[Phishing] Example 1", PHISHING_EXAMPLE_1),
                    ("[Phishing] Example 2", PHISHING_EXAMPLE_2),
                    ("[Phishing] Example 3", PHISHING_EXAMPLE_3),
                    ("[Normal] Example 1", NORMAL_EXAMPLE_1),
                    ("[Normal] Example 2", NORMAL_EXAMPLE_2),
                    ("[Normal] Example 3", NORMAL_EXAMPLE_3),
                ],
                label="예시 선택 (선택 시 아래 입력창에 자동 채워집니다)",
                value=None,
            )

            email_input = gr.Textbox(
                lines=12,
                label="이메일 본문 입력",
                placeholder="여기에 이메일 본문(body) 내용을 직접 입력하거나, 위 드롭다운에서 예시를 선택하세요.",
            )

            predict_button = gr.Button("피싱 여부 예측하기")

        with gr.Column():
            label_output = gr.Textbox(label="예측 라벨 (정상/피싱)")
            proba_output = gr.Number(label="피싱 확률 (1에 가까울수록 피싱 가능성 높음)")

    def fill_example(example_text: str):
        # 드롭다운에서 선택한 예시를 입력창에 넣어줌
        return example_text or ""

    example_dropdown.change(
        fn=fill_example,
        inputs=example_dropdown,
        outputs=email_input,
    )

    predict_button.click(
        fn=predict_phishing,
        inputs=email_input,
        outputs=[label_output, proba_output],
    )


if __name__ == "__main__":
    demo.launch()

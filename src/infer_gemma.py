# (Model Ver.2.3) Gemma 오픈모델
# Few-shot + COT를 활용해서 0.3B 모델로 분류 + 이메일의 본문 뿐만 아니라 제목도 input으로 활용
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # ✅ CSV 저장용


df = pd.read_csv("./CEAS_08.csv")

# (Model Ver.2.2) Gemma 오픈모델
# Few-shot + COT를 활용해서 분류
email_text = None
PROMPT_TEMPLATE = """
당신은 이메일 보안 분류 전문가입니다.
아래 이메일이 피싱 이메일인지 여부를 판단하세요.

아웃풋 포맷은 아래와 같습니다.
왜 피싱인지 또는 왜 피싱이 아닌지에 대한 간단한 한국어 설명 /// 1(피싱) 또는 0(정상)

# 예시1
Input:On Oct 19 2007, Rafael Garcia-Suarez wrote: > > Thanks, applied as #32141. The attached patch adds the suggested macro to avoid the extra function call when the operand is not a reference. No point penalizing everyone just for overload. -- Rick Delaney hfvo@bort.ca
Output:이 이메일은 Perl 관련 기능 개선 패치와 매크로 적용에 대해 개발자들이 기술적으로 논의하는 내용입니다. 로그인 정보, 비밀번호, 계좌 정보 요청이나 악성 링크 유도 등 피싱 요소가 전혀 없어 정상적인 기술 메일로 판단됩니다.///0

# 예시2
Input:The preparation we propose will help you to win a competition of male dimensions! Elongate your love weapon and improve your self-confidence! http://placefall.com/ to my financial interests, or vice versa. But this has been modified duethat whatever regulations they introduce do not do more harm thanSoros.
Output:이 이메일은 남성 신체 확대를 미끼로 한 스팸성 광고 문구와 함께 의심스러운 링크를 포함하고 있습니다. 긴박하거나 과장된 표현으로 사용자를 유혹하며, 악성 사이트로 유도할 가능성이 높으므로 피싱 이메일로 판단됩니다.///1

Input:{email_text}
Output:"""

model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)


def classify_email_llama(email_text):
    prompt = PROMPT_TEMPLATE.replace("{email_text}", email_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 모델 출력에서 마지막 줄만 추출
    if "///" not in response:
        return None, None  # 실패한 경우

    explanation, label_str = response.split("///")[-2], response.split("///")[-1]

    # 숫자만 추출
    label_str = label_str.strip()
    if label_str.startswith("1"):
        return explanation.strip(), 1
    elif label_str.startswith("0"):
        return explanation.strip(), 0
    else:
        return explanation.strip(), None


# ===================================================
# 4. 예측 + CSV 저장용 결과 수집
#    👉 subject + body를 concat 해서 input으로 사용
# ===================================================

num_sample = 10

# 가정: test_df 또는 df_test 같은 이름의 DataFrame에
# 'subject', 'body', 'label' 컬럼이 있음
# 예시: test_df = df_test  이런 식
X_test = df[["subject", "body"]].head(num_sample).values.tolist()
y_true = df["label"].head(num_sample).values

y_pred = []
rows_for_csv = []  # ✅ CSV에 저장할 행들을 담는 리스트

for (subject, body), true_label in tqdm(list(zip(X_test, y_true)), total=len(X_test)):
    # 제목 + 본문을 하나의 입력으로 합침
    combined_text = f"Subject: {subject}\n\nBody: {body}"

    explanation, pred = classify_email_llama(combined_text)

    # 실패한 경우 0으로 처리
    if pred is None:
        pred = 0

    y_pred.append(pred)

    # ✅ CSV에 들어갈 정보 저장
    rows_for_csv.append(
        {
            "subject": subject,  # 제목
            "body": body,  # 본문
            "input_text": combined_text,  # 모델 입력으로 사용한 concat 텍스트
            "cot": explanation,  # 모델의 설명(cot)
            "pred_label": int(pred),  # 예측 레이블
            "true_label": int(true_label),  # 실제 정답 레이블
        }
    )

y_pred = np.array(y_pred)

# DataFrame으로 만들어서 CSV로 저장
df_results = pd.DataFrame(rows_for_csv)
df_results.to_csv(
    "llama_phishing_results_with_subject_body_cot.csv",
    index=False,
    encoding="utf-8-sig",
)
print("✅ CSV 저장 완료: llama_phishing_results_with_subject_body_cot.csv")

# ============================================
# 5. 평가 지표 계산
# ============================================
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, zero_division=1)

print(f"\nAccuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# ============================================
# 6. Confusion Matrix 시각화
# ============================================
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Gemma phishing classifier with subject+body)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

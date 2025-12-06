# (Model Ver.2.2) Gemma 오픈모델
# Few-shot + COT를 활용해서 분류

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # ✅ CSV 저장용


df = pd.read_csv("../data/CEAS_08.csv")

X = df["body"].astype(str).values  # 이메일 본문
y = df["label"].values  # 0/1 라벨 (정상/피싱)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


email_text = None
PROMPT_TEMPLATE = """
당신은 이메일 보안 분류 전문가입니다.
아래 이메일이 피싱 이메일인지 여부를 판단하세요.

# 🔍 피싱 이메일 판별 핵심 체크리스트

## 1) 보낸 주소(Sender Email) 확인
- 도메인이 공식 도메인인지 확인  
  - 예: `@google.com` vs `@google-security-alerts.com`(가짜)
- 철자 하나 틀린 도메인도 의심  
  - `micros0ft.com`, `g00gle-alert.com` 등

---

## 2) 긴급성을 강조하는 문구
피싱 메일의 70%는 이런 문구로 시작함:
- “결제가 곧 중단됩니다”
- “계정이 정지될 예정입니다”
- “즉시 로그인해서 조치를 취하세요”

👉 사람을 당황시켜 바로 클릭하게 만드는 전략임.

---

## 3) 정보 업데이트 요구 / 로그인 요구
- “계정 정보를 업데이트하라”
- “비밀번호를 다시 입력하라”
- “결제 정보를 재등록하라”

기업은 이메일로 **로그인/결제 정보 요구를 거의 하지 않음.**

---

## 4) 링크 주소(URL) 실제 확인
링크 텍스트는 정상처럼 보이나, 실제 주소는 다를 수 있음.
- 마우스를 올려서(URL Hover) 실제 주소 확인  
- 단축 URL, 생소한 도메인 → 의심
- HTTPS라도 안전 보장은 아님 (`https://fake-google-login.com` 가능)

---

## 5) 맞춤형 정보가 없음
공식 이메일은 보통:
- 이름,
- 계정 ID,
- 결제 내역

등을 포함함.  
피싱 메일은 보통 “Dear user”, “Dear customer”처럼 **범용 호칭** 사용.

---

## 6) 어색한 문장, 문법 오류
- 영어 문장이 부자연스럽거나  
- 문법/띄어쓰기 오류가 많으면  
피싱 가능성 증가.

---

## 7) 수상한 첨부파일
- `.zip`, `.exe`, `.scr`, `.html`, `.htm` 파일 → 즉시 의심  
- 금융/배송/결제 관련인데 HTML 첨부파일? → 95% 피싱

---

## 8) 비정상적인 이메일 포맷
피싱/스팸 메일에서 자주 보이는 패턴:
- 지나치게 많은 `------` 구분선
- 의미 없는 공백
- 깨진 HTML 포맷

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
# 4. 예측 + JSONL 저장용 결과 수집
# ===================================================
import json

num_sample = 100
y_true = y_test
X_test, y_true = X_test[:num_sample], y_true[:num_sample]

y_pred = []
rows_for_jsonl = []  # ✅ JSONL에 저장할 행들을 담는 리스트

for text, true_label in tqdm(list(zip(X_test, y_true)), total=len(X_test)):
    explanation, pred = classify_email_llama(text)

    # 실패한 경우 0으로 처리
    if pred is None:
        pred = 0

    y_pred.append(pred)

    # ✅ JSONL에 들어갈 정보 저장
    rows_for_jsonl.append(
        {
            "input_text": text,
            "cot": explanation,
            "pred_label": int(pred),
            "true_label": int(true_label),
        }
    )

y_pred = np.array(y_pred)
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, zero_division=1)

print(f"\nAccuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# JSONL 파일로 저장
jsonl_path = "results_gemini_inference_subject_check.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for row in rows_for_jsonl:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ JSONL 저장 완료: {jsonl_path}")

from fig_utils import save_confusion_matrix

fig_path = "../fig/conf_matrix_gemini_subject_check.png"
save_confusion_matrix(conf_matrix, fig_path)

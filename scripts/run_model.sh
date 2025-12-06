pip3 install -r ../requirements.txt

# echo "logistic . . ."
# python3 ../src/model_logistic.py > ../log/model_logistic.log

# body가 NL 형태라 임베딩하여 분류할 수 있는 BERT 모델을 가장 먼저 활용하게 됨.
# 자원, 시간 리소스를 고려했을때 학습이 된 open 모델을 활용하는 게 최적이라 판단함
echo "logistic bert . . ."
python3 ../src/model_bert.py > ../log/model_bert.log
# 장점은 0.87로 꽤 괜찮은 정확도를 얻음
# 성능 수치를 개선하고자 LLM 활용을 고려함

# sLLM을 활용해 성능을 개선하려함
# 작은 사이즈의 open gemini 모델을 활용하여 분류 시도
echo "logistic gemini . . ."
python3 ../src/model_gemini.py > ../log/model_gemini.log
# 성능이 오히려 떨어짐. 정성평가를 통해 모델 판단에 설명이 부족하고 프롬프트를 명확히 하는바가 필요하다 판단함.
# 성능을 개선하고자 CoT + Few-shot 기법 활용 및 프롬프트 개선, subject도 input으로 활용


# echo "logistic gemini subject . . ."
# python3 ../src/model_gemini_subject.py > ../log/model_gemini_subject.log

# 몇 가지 개선하여 다시 sLLM 활용
echo "subject gemini subject check. . ."
python3 ../src/model_gemini_subject_check.py > ../log/model_gemini_subject_check.log
# 성능은 개선 되었으나 inference 시간이 너무나 오래 걸림

# infer 시간을 개선하기 위해 아예 EDA한 바를 토대로 Logistic Regression 모델을 활용하고자함
echo "logistic eda . . ."
python3 ../src/model_logistic_eda.py > ../log/model_logistic_eda.log
# 높은 성능 달성, 낮은 inference 시간 달성


# 전체 insight
# 주어진 Task에 따라서 꼭 LLM이 가장 좋은 성능을 내는것을 아님
# gemini에서 prompt engineering 만으로도 성능을 꽤 개선할 수 있는 점을 확인함
# 바이브 코딩 및 skywalk 등의 도구를 활용하여 자동화로 작업 시간을 많이 단축할 수 있었음
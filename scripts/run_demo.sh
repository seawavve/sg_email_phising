pip3 install -r ../requirements.txt

echo "logistic eda . . ."
python3 ../src/model_logistic_eda.py > ../log/model_logistic_eda.log

echo "demo . . ."
python3 ../src/demo.py
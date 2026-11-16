# ML·DL 기반 가짜 리뷰 탐지 모델 비교

Yelp 레스토랑 리뷰에서 가짜 리뷰를 가려내는 머신러닝·딥러닝 모델을 비교한 수업 팀 프로젝트입니다.

- 과목: 경희대학교 빅데이터의사결정분석 (2024년 1학기, A조)
- 기간: 2024.03 ~ 2024.06
- 팀: 김소륜, 우수현, 정윤서, 진유린 (4인)

## 분석 흐름

1. 정상 리뷰와 가짜 리뷰의 수를 맞춘 데이터셋 구성
2. 텍스트 전처리 후 Word2Vec 임베딩 평균과 수치형 변수(리뷰 길이, 평점, 극단성 등)를 결합
3. 머신러닝 10종과 딥러닝 3종(CNN, CNN-LSTM, CNN-RNN) 학습, 5회 반복 평균으로 비교
4. Permutation Importance로 수치형 변수 선택
5. 정상·가짜 리뷰의 단어 빈도, TF-IDF, 품사, 단어 길이 비교

## 결과 (5회 평균)

| 모델 | Accuracy | F1-score |
|---|---|---|
| Gradient Boosting | 0.807 | 0.812 |
| Random Forest | 0.791 | 0.800 |
| XGBoost | 0.794 | 0.798 |
| CNN-LSTM | 0.731 | 0.729 |
| CNN | 0.729 | 0.728 |

트리 기반 앙상블이 딥러닝 모델보다 높았습니다. 전체 수치는 `results/`에 있습니다.

## 내 역할 (김소륜)

- 일정과 역할 조율, 교수·조교 소통
- 베이스라인 머신러닝 모델링과 텍스트(TF-IDF) 단독 모델 평가
- 정상·가짜 리뷰의 품사와 단어 길이 비교 분석
- Permutation Importance 기준 상위 5개 수치형 변수 채택안 제안
- 발표 자료 구성과 검수, 최종 제출

임베딩과 딥러닝 모델, 최종 성능 산출은 우수현, 전처리와 성능 비교 시각화는 진유린, 수치형 머신러닝 코드와 참고문헌은 정윤서가 맡았습니다. 이 레포의 노트북은 팀 최종 제출본입니다.

## 노트북

| 파일 | 내용 |
|---|---|
| `notebooks/01_dataset_preprocessing.ipynb` | 라벨 균형 데이터셋 생성 |
| `notebooks/02_text_visualization.ipynb` | 단어 빈도, TF-IDF, 동시 출현, 품사 비교 |
| `notebooks/03_machine_learning_models.ipynb` | 머신러닝 10종 |
| `notebooks/04_deep_learning_models.ipynb` | CNN, CNN-LSTM, CNN-RNN |
| `notebooks/05_performance_visualization.ipynb` | 평균·표준편차 시각화 |
| `notebooks/06_performance_comparison.ipynb` | 모델 성능 비교 |

## 데이터

Yelp 레스토랑 리뷰 데이터는 재배포할 수 없어 포함하지 않았습니다. 노트북에 남아 있던 리뷰 원문 출력은 지웠습니다.

## 환경

Python 3 (Google Colab), pandas, scikit-learn, xgboost, gensim, tensorflow, nltk, wordcloud

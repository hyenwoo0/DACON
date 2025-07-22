# 📘 생성형 AI(LLM)와 인간: 텍스트 판별 챌린지

본 프로젝트는 DACON에서 주최한 **“생성형 AI(LLM)와 인간: 텍스트 판별 챌린지”** 참가를 위한 코드 저장소입니다.  
문단 단위(Paragraph)의 텍스트가 사람(Human)이 작성한 것인지, 생성형 AI가 작성한 것인지 예측하는 AI 모델을 개발하는 것을 목표로 합니다.

---

## 🧩 문제 설명

- 문단(Paragraph) 단위의 텍스트가 사람(0) 혹은 AI(1)에 의해 작성된 것인지 판별
- 예측값은 **0~1 사이의 확률**로 출력
- **학습 데이터**는 글(Full Text) 단위의 라벨만 제공되며, 일부 문단만 AI가 작성된 경우도 전체 글은 `1`로 라벨링됨
- **평가 데이터**는 **문단 단위**이며, 각 문단이 AI에 의해 작성되었을 확률을 예측해야 함
- 동일한 `title`을 가진 문단은 동일한 글에 속하므로, 글 내 문단 간의 **상호 참조 가능**

---

## 🗂️ 폴더 구조 및 주요 코드 설명

DACON/
├── balanced_data_paragraph.py # 0과 1의 데이터 불균형을 under-sampling을 활용해 해결

├── data.py # 데이터 불러오기 및 데이터의 탐색

├── data_arg.py # 데이터를 증강시키려는 시도(실패)

├── deberta_train_under.py # DeBERTa 모델 학습 (under-sampling 데이터를 통한 학습)

├── ex.ipynb # 실험용 노트북

├── filter_data.py # 문단 단위로 필터링하여 데이터 증강(실패)

├── tf-idf_train.py # TF-IDF 기반 학습 코드(under-sampling 데이터를 통한 학습)

├── tf.py # 대회에서 제공한 베이스 코드를 활용한 tf-idf 학습

├── train.py # 기본 학습 코드(deberta의 기본 학습 코드)

├── train_kcelectra.py # KoELECTRA 기반 학습 코드

├── weighted_average.py # 모델 앙상블 (가중 평균) (under-sampling 데이터를 통한 학습) (deberta_train_under.py, tf-idf_train.py, train_kcelectra.py 세가지 모델의 가중 평균)

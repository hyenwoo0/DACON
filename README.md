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

## 📂 Dataset 정보

###  `train.csv` – 학습 데이터

| 컬럼명      | 설명                                  |
|-------------|---------------------------------------|
| `title`     | 글의 제목                             |
| `full_text` | 전체 글 텍스트                        |
| `generated` | 해당 글이 AI로 생성되었는지 여부 (0: 사람, 1: AI) |

※ 주의: 문단 단위 라벨은 제공되지 않으며, 글 전체 기준으로 라벨링됨  
(일부 문단만 AI 작성이어도 전체 글 라벨은 `1`)

---

###  `test.csv` – 평가 데이터

| 컬럼명             | 설명                              |
|--------------------|-----------------------------------|
| `ID`               | 평가 샘플 고유 식별자             |
| `title`            | 글의 제목                         |
| `paragraph_index`  | 글을 구성하는 문단의 번호 (0부터 시작) |
| `paragraph_text`   | 문단 텍스트                       |

※ 같은 `title`을 가진 문단은 동일한 글에 속하며,  
이를 활용한 문단 간 정보 공유(추론)는 **허용**됩니다.

---

###  `sample_submission.csv` – 제출 양식

| 컬럼명      | 설명                                  |
|-------------|---------------------------------------|
| `ID`        | 평가 샘플 고유 식별자                 |
| `generated` | 해당 문단이 AI가 작성했을 확률 (0~1) |

→ 이 형식에 맞춰 `submission_*.csv` 파일을 생성하여 제출


## 🔬 실험 순서

본 프로젝트는 다음과 같은 순서로 실험을 진행하였습니다:

1. **데이터 탐색 및 전처리**
   - `data.py`를 활용하여 데이터 로딩 및 구조 분석
   - 라벨 불균형 확인 후, under-sampling 적용 (`balanced_data_paragraph.py`)

2. **모델 학습**
   - DeBERTa 모델 학습 (`deberta_train_under.py`)
   - KoELECTRA 모델 학습 (`train_kcelectra.py`)
   - TF-IDF + 분류기 학습 (`tf-idf_train.py`)

3. **실험 분석**
   - 각 모델의 출력 결과를 저장하고 성능 비교
   - 모델 별 특징 및 한계점 도출

4. **앙상블 및 최종 제출 파일 생성**
   - `weighted_average.py`를 통해 세 모델의 예측값을 가중 평균
   - 최종 제출 파일 생성

5. **실패한 시도**
   - `data_arg.py`: 데이터 증강 실험 (효과 미미)
   - `filter_data.py`: 문단 필터링을 통한 성능 향상 시도 (성과 없음)


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


## 🧪 실험 결과 (Macro F1 기준)

| 제출 파일명                                 | 시간                     | 점수          | 설명                                      |
|------------------------------------------|-------------------------|--------------|----------------------------------------- |
| submission_deberta_augmented_sampled.csv | 2025-07-10 13:57:35     | 0.6034       | 증강 + 샘플링한 DeBERTa 학습 결과              |
| submission_deberta_balanced.csv          | 2025-07-10 15:40:13     | 0.7607       | Under-sampling한 DeBERTa 모델              |
| submission_electra_balanced.csv          | 2025-07-10 16:36:22     | 0.7770       | Under-sampling한 KoELECTRA 모델            |
| submission_tfidf_lr.csv                  | 2025-07-10 17:07:10     | 0.6507       | Under-sampling한 TF-IDF + LogisticReg.    |
| **submission_ensemble_weighted.csv**     | **2025-07-10 17:11:26** | **0.7999**   | 세 모델 앙상블 (가중 평균)                     |

위 표는 제출한 각 모델의 결과 파일명, 제출 시간, 그리고 ROC-AUC 기준 성능을 정리한 것입니다.  
**가중 평균 앙상블 모델**(`weighted_average.py`)이 가장 높은 성능인 **0.7999**를 기록했습니다.  
단일 모델 중에서는 **KoELECTRA 기반 모델**이 가장 높은 성능(0.7770)을 보였습니다.



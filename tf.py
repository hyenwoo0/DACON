import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ✅ lambda 대신 함수 정의 (pickle 가능)
def extract_title(x):
    return x['title']

def extract_text(x):
    return x['full_text']

print("✅ [1] 데이터 불러오는 중...")
train = pd.read_csv('/Users/johyeon-u/PycharmProjects/DACON/open/train.csv', encoding='utf-8-sig')
test = pd.read_csv('/Users/johyeon-u/PycharmProjects/DACON/open/test.csv', encoding='utf-8-sig')
print("✅ [1-1] 데이터 불러오기 완료!")

print("✅ [2] 학습/검증 데이터 분할 중...")
X = train[['title', 'full_text']]
y = train['generated']
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print("✅ [2-1] 데이터 분할 완료!")

print("✅ [3] TF-IDF 벡터라이저 설정 중...")
get_title = FunctionTransformer(extract_title, validate=False)
get_text = FunctionTransformer(extract_text, validate=False)

vectorizer = FeatureUnion([
    ('title', Pipeline([
        ('selector', get_title),
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000))
    ])),
    ('full_text', Pipeline([
        ('selector', get_text),
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000))
    ]))
])
print("✅ [3-1] 벡터라이저 설정 완료!")

print("✅ [4] 학습 데이터 벡터화 중...")
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print("✅ [4-1] 벡터화 완료!")

print("✅ [5] 모델 학습 중...")
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_vec, y_train)
print("✅ [5-1] 모델 학습 완료!")

# ✅ 모델과 벡터라이저 저장
print("✅ [5-2] 모델 및 벡터라이저 저장 중...")
joblib.dump(xgb, 'tf/xgb_model.pkl')
joblib.dump(vectorizer, 'tf/vectorizer.pkl')
print("✅ [5-3] 저장 완료! (xgb_model.pkl, vectorizer.pkl)")

print("✅ [6] 검증 데이터 예측 및 AUC 계산 중...")
val_probs = xgb.predict_proba(X_val_vec)[:, 1]
auc = roc_auc_score(y_val, val_probs)
print(f"✅ [6-1] Validation AUC: {auc:.4f}")

print("✅ [7] 테스트 데이터 전처리 중...")
test = test.rename(columns={'paragraph_text': 'full_text'})
X_test = test[['title', 'full_text']]
X_test_vec = vectorizer.transform(X_test)
print("✅ [7-1] 테스트 데이터 벡터화 완료!")

print("✅ [8] 테스트 데이터 예측 중...")
probs = xgb.predict_proba(X_test_vec)[:, 1]
print("✅ [8-1] 예측 완료!")

print("✅ [9] 결과 저장 중...")
sample_submission = pd.read_csv('/Users/johyeon-u/PycharmProjects/DACON/open/sample_submission.csv', encoding='utf-8-sig')
sample_submission['generated'] = probs
sample_submission.to_csv('./baseline_submission.csv', index=False)
print("🎉 [9-1] 결과 저장 완료! 제출 준비 완료!")

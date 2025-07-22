import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
train_path = "/home/user/DACON/open/train_paragraph_balanced.csv"
test_path = "/home/user/DACON/open/test.csv"
submission_path = "/home/user/DACON/open/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. 입력 텍스트 생성 (title + paragraph_text)
train_df["input_text"] = train_df["title"] + " " + train_df["paragraph_text"]
test_df["input_text"] = test_df["title"] + " " + test_df["paragraph_text"]

# 3. TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(train_df["input_text"])
y = train_df["generated"].astype(int)

X_test = vectorizer.transform(test_df["input_text"])

# 4. 검증 데이터로 split (optional: 평가용)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# 5. 분류기 학습 (로지스틱 회귀)
clf = LogisticRegression(max_iter=300, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 6. 검증 (optional)
y_pred = clf.predict(X_valid)
y_pred_proba = clf.predict_proba(X_valid)[:,1]
print(f"Valid Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
print(f"Valid ROC AUC: {roc_auc_score(y_valid, y_pred_proba):.4f}")

# 7. 테스트 예측
test_proba = clf.predict_proba(X_test)[:,1]

# 8. 제출 파일 생성
submission = pd.read_csv(submission_path)
submission["generated"] = test_proba
submission.to_csv("submission_tfidf_lr.csv", index=False)
print("[INFO] 제출 파일 생성 완료: submission_tfidf_lr.csv")

import pandas as pd

deberta = pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/submission_deberta_balanced.csv")
electra = pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/submission_electra_balanced.csv")
tfidf = pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/submission_tfidf_lr.csv")

# 가중 평균
ensemble_probs = (
    0.5 * electra["generated"] +
    0.4 * deberta["generated"] +
    0.1 * tfidf["generated"]
)

submission = electra.copy()
submission["generated"] = ensemble_probs
submission.to_csv("submission_ensemble_weighted.csv", index=False)
print("[INFO] 가중치 앙상블 제출 파일 생성 완료!")

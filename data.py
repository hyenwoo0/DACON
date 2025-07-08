import pandas as pd
import numpy as np

# ———————— train.csv (문서 단위) EDA ————————
df_train = pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/open/train.csv")

# 1) 문서 수
num_docs = df_train.shape[0]

# 2) 라벨 분포
label_counts = df_train["generated"].value_counts()
label_ratio  = df_train["generated"].value_counts(normalize=True)

# 3) 전체 글 길이(문자 수) 통계
df_train["char_count"] = df_train["full_text"].str.len()
train_length_stats = df_train["char_count"].describe()

print("=== train.csv EDA ===")
print(f"문서 수: {num_docs}")
print("라벨 분포:\n", label_counts.to_dict())
print("라벨 비율:\n", label_ratio.round(3).to_dict())
print("전체 글(문서) 길이 통계:\n", train_length_stats.to_dict(), "\n")


# ———————— test.csv (문단 단위) EDA ————————
chunksize = 100_000
doc_titles = set()
paras_per_doc = []
para_lengths = []

for chunk in pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/open/test.csv", chunksize=chunksize):
    # 문서 제목 집계
    doc_titles.update(chunk["title"].unique())
    # 문단 수 (문서별)
    cnt = chunk.groupby("title")["paragraph_index"].nunique()
    paras_per_doc.extend(cnt.tolist())
    # 문단 텍스트 길이
    para_lengths.extend(chunk["paragraph_text"].str.len().tolist())

print("=== test.csv EDA ===")
print(f"유니크 문서 수: {len(doc_titles)}")
print(f"문단 수 분포 (min/max/mean): "
      f"{np.min(paras_per_doc)}/"
      f"{np.max(paras_per_doc)}/"
      f"{np.mean(paras_per_doc):.2f}")
print(f"문단 길이(문자 수) 통계:\n{pd.Series(para_lengths).describe().to_dict()}")

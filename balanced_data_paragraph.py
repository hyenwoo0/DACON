import pandas as pd

# 문단 단위 학습 데이터 로드
df = pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/train_paragraph.csv")

# 클래스별 개수 확인
count_ai = df[df['generated'] == 1].shape[0]
count_human = df[df['generated'] == 0].shape[0]

print(f"AI 문단 수: {count_ai}, 사람 문단 수: {count_human}")

# 사람 문단을 AI 문단 수만큼 랜덤 샘플링
human_sampled = df[df['generated'] == 0].sample(n=count_ai, random_state=42)

# AI 문단과 결합
balanced_df = pd.concat([human_sampled, df[df['generated'] == 1]])

# 셔플
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 결과 확인
print(balanced_df['generated'].value_counts())

# 저장
balanced_df.to_csv("train_paragraph_balanced.csv", index=False)

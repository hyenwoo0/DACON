import pandas as pd

# 학습 데이터 로드
train_df = pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/open/train.csv")

# 문단 단위로 분할
paragraphs = []
for idx, row in train_df.iterrows():
    full_text = row['full_text']
    label = row['generated']
    title = row['title']

    # 문단 분리 (빈 줄 기준)
    split_paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]

    for i, para in enumerate(split_paragraphs):
        paragraphs.append({
            "title": title,
            "paragraph_index": i,
            "paragraph_text": para,
            "generated": label
        })

# 새 DataFrame 생성
paragraph_df = pd.DataFrame(paragraphs)

# 확인
print(paragraph_df.head())
print(paragraph_df['generated'].value_counts())

# 저장
paragraph_df.to_csv("train_paragraph.csv", index=False)

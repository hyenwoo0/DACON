from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

# 1. 데이터 로딩 및 분할
df = pd.read_csv("/Users/johyeon-u/PycharmProjects/DACON/open/train.csv")
X_train, X_val, y_train, y_val = train_test_split(df["full_text"], df["generated"], test_size=0.2, random_state=42)

train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
val_ds = Dataset.from_dict({"text": X_val.tolist(), "label": y_val.tolist()})

# 2. 전처리 함수
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

# 3. 평가 지표
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 4. 모델/토크나이저 불러오기
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5. 데이터 토크나이즈
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# 6. 훈련 설정
training_args = TrainingArguments(
    output_dir="./deberta_result",
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# 7. 학습 시작
trainer.train()

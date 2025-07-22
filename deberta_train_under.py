# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ——— 파일 및 모델 경로 설정 ———
train_path        = "/home/user/DACON/open/train_paragraph_balanced.csv"  # 여기를 바꿔주세요
test_path         = "/home/user/DACON/open/test.csv"
submission_path   = "/home/user/DACON/open/sample_submission.csv"
model_ckpt        = "microsoft/deberta-v3-base"
output_dir        = "./deberta_balanced"
cache_dir         = "./cache_tokenized"

# ——— 1. 데이터 로드 ———
df = pd.read_csv(train_path)
df["label"] = df["generated"].astype(int)  # label 컬럼 생성

# ——— 2. Train/Valid Stratified Split ———
train_df, valid_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)
print(f"[INFO] Train: {len(train_df)}, Valid: {len(valid_df)}")

# ——— 3. Dataset 객체 생성 ———
train_ds = Dataset.from_pandas(train_df.drop(columns=["generated"]))
valid_ds = Dataset.from_pandas(valid_df.drop(columns=["generated"]))
test_df  = pd.read_csv(test_path)
test_ds  = Dataset.from_pandas(test_df)

# ——— 4. Tokenizer & preprocess 정의 ———
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
def preprocess(batch):
    texts = [t + " " + p for t, p in zip(batch["title"], batch["paragraph_text"])]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

# ——— 5. Train/Valid 데이터 토크나이징 ———
num_proc = min(2, os.cpu_count() // 2)
train_tokenized = train_ds.map(
    preprocess,
    batched=True,
    batch_size=256,
    num_proc=num_proc,
    remove_columns=["title", "paragraph_index", "paragraph_text"]
).rename_column("label", "labels")

valid_tokenized = valid_ds.map(
    preprocess,
    batched=True,
    batch_size=256,
    num_proc=num_proc,
    remove_columns=["title", "paragraph_index", "paragraph_text"]
).rename_column("label", "labels")

# ——— 6. 테스트 데이터 토크나이징 ———
test_ds = test_ds.map(
    preprocess,
    batched=True,
    batch_size=256,
    num_proc=1,
    remove_columns=["title", "paragraph_index", "paragraph_text"]
)

# ——— 7. 모델 & Trainer 설정 ———
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_roc_auc",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=42,
    ddp_find_unused_parameters=False,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "roc_auc":  roc_auc_score(labels, probs),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ——— 8. 학습 ———
trainer.train()

# ——— 9. 모델 & 토크나이저 저장 ———
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# ——— 10. 추론 & 제출 파일 생성 ———
preds = trainer.predict(test_ds)
probs = F.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
submission = pd.read_csv(submission_path)
submission["generated"] = probs
submission.to_csv("submission_deberta_balanced.csv", index=False)
print("[INFO] 제출 파일 생성 완료: submission_deberta_balanced.csv")

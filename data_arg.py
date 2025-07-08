import argparse
import os
import pandas as pd
import random
import time
import numpy as np
from transformers import pipeline

# AI 백-트랜슬레이션 증강 스크립트 (Mac CPU용)
# Hugging Face CLI 로그인 후 실행하세요: `huggingface-cli login`

def chunk_text(text: str, chunk_size: int):
    """텍스트를 최대 chunk_size 글자 단위로 분할"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def back_translate(text: str, ko_en, en_ko, max_retries: int = 3):
    """한→영→한 백-트랜슬레이션, 오류 시 재시도"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[BT] 번역 시도: {attempt}/{max_retries}")
            english = ko_en(text, max_length=512)[0]['translation_text']
            korean = en_ko(english, max_length=512)[0]['translation_text']
            return korean
        except Exception as e:
            print(f"[BT] 오류 발생: {e}, 재시도 중...")
            time.sleep(1)
    # 최대 재시도 후 실패 시 원본 텍스트 사용
    print("[BT] 경고: 백-트랜슬레이션 실패, 원본 텍스트 사용")
    return text

def augment_ai(df_ai: pd.DataFrame, n_to_augment: int, ko_en, en_ko, chunk_size: int):
    """AI 샘플을 백-트랜슬레이션으로 증강"""
    augmented = []
    indices = list(df_ai.index)
    print(f"[Augment] 증강 대상 AI 샘플 수: {n_to_augment}")
    for i in range(n_to_augment):
        if i % 10 == 0:
            print(f"[Augment] 진행: {i}/{n_to_augment}")
        idx = random.choice(indices)
        text = df_ai.at[idx, 'full_text']
        chunks = chunk_text(text, chunk_size)
        translated_chunks = [back_translate(c, ko_en, en_ko) for c in chunks]
        augmented.append({
            'title':     f"{df_ai.at[idx, 'title']}_aug",
            'full_text': " ".join(translated_chunks),
            'generated': 1
        })
    print("[Augment] 증강 완료")
    return pd.DataFrame(augmented)

def main():
    # 재현성 확보
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="AI 백-트랜슬레이션 증강 스크립트 (Mac CPU용)")
    parser.add_argument(
        '--train_csv', type=str, default='/Users/johyeon-u/PycharmProjects/DACON/open/train.csv',
        help='원본 train.csv 경로'
    )
    parser.add_argument(
        '--out_csv', type=str, default='train_balanced_bt.csv',
        help='증강 결과 저장 경로'
    )
    parser.add_argument(
        '--chunk_size', type=int, default=400,
        help='텍스트 청크 사이즈 (문자 수, default: %(default)s)'
    )
    args = parser.parse_args()

    print(f"[Init] train_csv: {args.train_csv}")
    print(f"[Init] out_csv:   {args.out_csv}")
    print(f"[Init] chunk_size: {args.chunk_size}")

    # 데이터 로드
    print("[Load] 데이터 로드 중...")
    df = pd.read_csv(args.train_csv)
    ai_df = df[df['generated'] == 1].reset_index(drop=True)
    human_df = df[df['generated'] == 0].reset_index(drop=True)
    print(f"[Load] 총 문서: {len(df)}, AI: {len(ai_df)}, Human: {len(human_df)}")

    n_to_aug = len(human_df) - len(ai_df)
    if n_to_aug <= 0:
        print("[Skip] 증강이 필요하지 않습니다. 이미 균형 상태입니다.")
        return

    # 번역 파이프라인 초기화
    print("[Pipeline] 번역 파이프라인 초기화 (Mac/CPU)...")
    ko_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en", device=-1)
    en_ko = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko", device=-1)
    print("[Pipeline] 파이프라인 준비 완료")

    # 증강 수행
    aug_df = augment_ai(ai_df, n_to_aug, ko_en, en_ko, args.chunk_size)

    # 결합 및 셔플
    print("[Combine] 증강본과 원본 결합 및 셔플링")
    balanced_ai = pd.concat([ai_df, aug_df], ignore_index=True)
    balanced_df = pd.concat([human_df, balanced_ai], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 저장
    print("[Save] 결과 저장 중...")
    balanced_df.to_csv(args.out_csv, index=False)
    print(f"[Save] 균형 데이터셋 저장 완료: {args.out_csv}")
    print("[Done] 최종 분포:", balanced_df['generated'].value_counts().to_dict())

if __name__ == '__main__':
    main()

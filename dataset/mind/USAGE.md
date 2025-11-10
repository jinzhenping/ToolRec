# MIND 데이터셋 사용 가이드

## 파일 구조

- `mind.train.inter`: 학습용 상호작용 데이터 (39,955개)
- `mind.test.inter`: 테스트용 상호작용 데이터 (제공된 groundtruth + negative 샘플, 5,000개)
- `mind.item`: 뉴스 정보 (제목, 카테고리, 서브카테고리)
- `mind.user`: 사용자 정보
- `mind_test_groundtruth.pkl`: 테스트 groundtruth 및 위치 정보

## 학습 및 테스트 실행

### 기본 명령어

```bash
python run_recbole.py --model=SASRec --dataset=mind --config_files=dataset/mind/mind.yaml
```

이 명령어는:
1. **학습**: `mind.train.inter`를 90% train, 10% valid로 분할하여 학습
2. **테스트**: `mind.test.inter` (제공된 테스트셋)을 사용하여 평가

### 다른 모델 사용

```bash
# BERT4Rec
python run_recbole.py --model=BERT4Rec --dataset=mind --config_files=dataset/mind/mind.yaml

# GRU4Rec
python run_recbole.py --model=GRU4Rec --dataset=mind --config_files=dataset/mind/mind.yaml
```

## 설정 설명

- `benchmark_filename: ['train', 'test']`: 별도의 train/test 파일 사용
- `split: {'RS': [0.9, 0.1]}`: train 데이터를 90% train, 10% valid로 분할
- `gpu_id: 1`: GPU 1번 사용

## 테스트 데이터 정보

- `mind.test.inter`에는 각 사용자별로:
  - 1개의 groundtruth (rating=1.0)
  - 4개의 negative 샘플 (rating=0.0)
  - 순서는 랜덤하게 섞여 있음 (순서 편향 방지)

- `mind_test_groundtruth.pkl`에는:
  - `groundtruth`: 사용자 ID → groundtruth 아이템 ID 매핑
  - `positions`: 사용자 ID → groundtruth의 위치 (0-4) 매핑

## 커스텀 평가

제공된 테스트셋의 groundtruth와 negative 샘플을 활용한 커스텀 평가를 원하면 `mind_test_groundtruth.pkl` 파일을 사용할 수 있습니다.


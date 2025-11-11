# GPU 설정 가이드

## 문제 상황
`mind.yaml`에 `gpu_id: 1`로 설정했지만, 실제로는 GPU 0번이 사용되고 있습니다.

## 해결 방법

### 방법 1: 명령줄에서 GPU ID 명시 (권장)

```bash
# 환경 변수로 GPU 1번만 보이게 설정
CUDA_VISIBLE_DEVICES=1 python run_recbole.py \
  --dataset=mind \
  --model=SASRec \
  --config_files=dataset/mind/mind.yaml \
  --gpu_id=1
```

또는:

```bash
# 환경 변수만 설정
export CUDA_VISIBLE_DEVICES=1
python run_recbole.py \
  --dataset=mind \
  --model=SASRec \
  --config_files=dataset/mind/mind.yaml
```

### 방법 2: GPU 0번 프로세스 확인 및 종료

```bash
# GPU 사용 중인 프로세스 확인
nvidia-smi

# 특정 프로세스 종료 (필요한 경우)
kill -9 <PID>
```

### 방법 3: Python 코드에서 직접 설정

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

## RecBole의 GPU 설정 방식

RecBole은 `CUDA_VISIBLE_DEVICES` 환경 변수를 사용하여 GPU를 선택합니다:
- `gpu_id: 1` → `CUDA_VISIBLE_DEVICES=1` 설정
- 이렇게 하면 물리적 GPU 1번이 논리적 GPU 0번으로 매핑됩니다
- PyTorch는 항상 `cuda:0`을 사용하지만, 실제로는 물리적 GPU 1번을 사용합니다

## 확인 방법

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

## 주의사항

- `CUDA_VISIBLE_DEVICES=1`을 설정하면, PyTorch는 GPU 0번으로 인식하지만 실제로는 물리적 GPU 1번을 사용합니다
- 여러 GPU를 사용하려면 `gpu_id: "0,1"` 형식으로 설정할 수 있습니다
- GPU 0번에 다른 프로세스가 실행 중이면 메모리 부족 오류가 발생할 수 있습니다


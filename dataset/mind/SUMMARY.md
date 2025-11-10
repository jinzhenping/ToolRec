# MIND 데이터셋 LLM-Empowered Recommendation 완료 현황

## ✅ 완료된 작업

1. **데이터셋 변환**
   - MIND 원본 데이터 → RecBole 형식 변환 완료
   - `mind.inter`, `mind.item`, `mind.user`, `mind.test.inter` 생성

2. **기본 추천 모델 학습**
   - 모델: SASRec
   - 저장 경로: `dataset/saved_file/SASRec-Nov-10-2025_16-08-05.pth`

3. **Attribute-Oriented Retrieval Tools 학습**
   - Category 기반: `SASRec_AddInfo2-Nov-10-2025_16-35-20.pth`
   - Subcategory 기반: `SASRec_AddInfo2-Nov-10-2025_16-54-08.pth`

4. **설정 파일 업데이트**
   - `utils.py`: 모델 파일 경로 등록 완료
   - `dataset/prompts/mind_pattern.json`: 프롬프트 패턴 생성 완료
   - `recbole/data/dataset/dataset.py`: MIND 데이터셋 지원 추가 완료

## 📋 다음 단계

### 1. 프롬프트 파일 생성 (필수)

LLM recommendation 실행을 위해 다음 파일들을 생성해야 합니다:

```bash
# 1. 테스트 사용자 히스토리 정보 생성
python run_recbole.py \
  --dataset=mind \
  --model=SASRec \
  --config_files=dataset/mind/mind.yaml \
  --gpu_id=1 \
  --dump_userInfo_chat=True \
  --test_v=test/

# 2. 사용자/아이템 프로필 생성
python run_recbole.py \
  --dataset=mind \
  --model=SASRec \
  --config_files=dataset/mind/mind.yaml \
  --gpu_id=1 \
  --dump_profile=True \
  --test_v=test/
```

### 2. ICL 예제 파일 생성 (선택)

`dataset/prompts/test/mind_ICL.json` 파일을 생성해야 합니다. 
현재는 `ml-1m_ICL.json`을 참고하여 MIND 데이터셋에 맞게 수정하여 생성할 수 있습니다.

### 3. LLM API 설정 확인

`chat_api.py`에서 OpenAI API 키와 엔드포인트를 설정해야 합니다:
- `openai.api_key`: API 키 설정
- `openai.api_base`: API 엔드포인트 설정

### 4. LLM-Empowered Recommendation 실행

```bash
python chat_RecAct.py \
  --start=0 \
  --step_num=100
```

## 📁 생성된 파일 목록

### 데이터 파일
- `dataset/mind/mind.inter` - 학습용 상호작용
- `dataset/mind/mind.train.inter` - 학습용 상호작용 (복사본)
- `dataset/mind/mind.test.inter` - 테스트용 상호작용 (groundtruth + negative)
- `dataset/mind/mind.item` - 뉴스 정보
- `dataset/mind/mind.user` - 사용자 정보
- `dataset/mind/mind_test_groundtruth.pkl` - 테스트 groundtruth 정보

### 설정 파일
- `dataset/mind/mind.yaml` - 기본 모델 설정
- `dataset/mind/mind_addinfo.yaml` - Attribute-Oriented Retrieval Tool 설정
- `dataset/prompts/mind_pattern.json` - 프롬프트 패턴

### 모델 파일
- `dataset/saved_file/SASRec-Nov-10-2025_16-08-05.pth` - 기본 모델
- `dataset/saved_file/SASRec_AddInfo2-Nov-10-2025_16-35-20.pth` - Category 기반 도구
- `dataset/saved_file/SASRec_AddInfo2-Nov-10-2025_16-54-08.pth` - Subcategory 기반 도구

## 🔧 주요 수정 사항

1. **RecBole 코드 수정**
   - `recbole/data/dataset/dataset.py`: MIND 데이터셋 지원 추가
   - `recbole/data/dataset/sequential_dataset.py`: Benchmark 모드에서 데이터 augmentation 보장
   - `recbole/data/utils.py`: Benchmark 파일 사용 시 train/valid 분할 처리
   - `recbole/trainer/trainer.py`: PyTorch 2.6+ 호환성 및 평가 로직 수정
   - `recbole/evaluator/collector.py`: Device mismatch 및 인덱스 범위 체크 추가
   - `recbole/evaluator/metrics.py`: Division by zero 처리
   - `recbole/model/sequential_recommender/*.py`: `torch.load`에 `weights_only=False` 추가

2. **데이터 변환 스크립트**
   - `convert_mind_to_recbole.py`: MIND 데이터셋 → RecBole 형식 변환

## 📚 참고 문서

- `dataset/mind/USAGE.md`: 데이터셋 사용 가이드
- `dataset/mind/NEXT_STEPS.md`: Attribute-Oriented Retrieval Tool 준비 가이드
- `dataset/mind/LLM_RECOMMENDATION_STEPS.md`: LLM-Empowered Recommendation 실행 가이드


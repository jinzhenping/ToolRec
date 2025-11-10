# MIND 데이터셋 다음 단계 가이드

## 현재 완료된 단계
✅ **1. Pretrain Sequential Recommendation**
- 모델: SASRec
- 저장 경로: `dataset/saved_file/SASRec-Nov-10-2025_16-08-05.pth`

## 다음 단계: Attribute-Oriented Retrieval Tool 준비

### 목적
아이템 속성(category, subcategory)을 활용하여 속성 기반 검색 도구를 구축합니다.
LLM이 "카테고리='스포츠'로 검색"과 같은 속성 조건으로 아이템을 검색할 수 있게 합니다.

### 단계별 실행

#### 1. Category 기반 검색 도구 학습

```bash
python run_recbole.py \
  --dataset=mind \
  --model=SASRec_AddInfo2 \
  --config_files=dataset/mind/mind_addinfo.yaml \
  --gpu_id=1 \
  --show_progress=False \
  --load_pretrain=True \
  --freeze_Rec_Params=True \
  --item_additional_usage=True \
  --item_additional_feature=category \
  --side_feature_size=100 \
  --bert_usage=True \
  --pretrained_name=/SASRec-Nov-10-2025_16-08-05.pth
```

**설명:**
- `--model=SASRec_AddInfo2`: 속성 정보를 활용하는 모델
- `--item_additional_feature=category`: category 속성 사용
- `--load_pretrain=True`: 사전 학습된 SASRec 모델 로드
- `--freeze_Rec_Params=True`: 기본 추천 모델 파라미터 고정 (속성 임베딩만 학습)

#### 2. Subcategory 기반 검색 도구 학습

```bash
python run_recbole.py \
  --dataset=mind \
  --model=SASRec_AddInfo2 \
  --config_files=dataset/mind/mind_addinfo.yaml \
  --gpu_id=1 \
  --show_progress=False \
  --load_pretrain=True \
  --freeze_Rec_Params=True \
  --item_additional_usage=True \
  --item_additional_feature=subcategory \
  --side_feature_size=100 \
  --bert_usage=True \
  --pretrained_name=/SASRec-Nov-10-2025_16-08-05.pth
```

#### 3. 학습 완료 후 utils.py 업데이트

학습이 완료되면 저장된 모델 파일명을 확인하고 `utils.py`의 `model_file_dict`를 업데이트하세요:

```python
'mind': {
    'None': 'SASRec-Nov-10-2025_16-08-05.pth',
    'category': 'SASRec_AddInfo2-Nov-10-2025_XX-XX-XX.pth',  # 실제 파일명으로 변경
    'subcategory': 'SASRec_AddInfo2-Nov-10-2025_XX-XX-XX.pth',  # 실제 파일명으로 변경
}
```

### 다음 단계 (3단계)
학습이 완료되면:
1. `utils.py`에서 `dataset_name="mind"` 설정 확인
2. `chat_RecAct.py` 실행하여 LLM-empowered recommendation 시작

## 참고사항

- 각 속성별로 별도의 모델을 학습합니다
- 기본 추천 모델(SASRec)의 파라미터는 고정하고, 속성 임베딩만 학습합니다
- 학습 시간은 속성 개수에 따라 다르지만, 일반적으로 기본 모델보다 빠릅니다


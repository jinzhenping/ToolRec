# LLM-Empowered Recommendation 실행 가이드

## 현재 완료된 단계

✅ **1. Pretrain Sequential Recommendation**
- 모델: SASRec
- 저장 경로: `dataset/saved_file/SASRec-Nov-10-2025_16-08-05.pth`

✅ **2. Attribute-Oriented Retrieval Tools**
- Category 기반: `SASRec_AddInfo2-Nov-10-2025_16-35-20.pth`
- Subcategory 기반: `SASRec_AddInfo2-Nov-10-2025_16-54-08.pth`

✅ **3. utils.py 업데이트 완료**
- 모델 파일 경로가 `model_file_dict`에 등록됨

## 다음 단계: 프롬프트 파일 생성

LLM-empowered recommendation을 실행하기 전에 사용자 프로필과 아이템 프로필을 생성해야 합니다.

### 1. 테스트 사용자 정보 파일 생성

먼저 테스트 사용자들의 히스토리 정보를 포함한 `uid_dict.pkl` 파일을 생성해야 합니다:

```bash
python run_recbole.py \
  --dataset=mind \
  --model=SASRec \
  --config_files=dataset/mind/mind.yaml \
  --gpu_id=1 \
  --dump_userInfo_chat=True \
  --test_v=test/
```

이 명령어는:
- 테스트 데이터에서 사용자 히스토리 정보를 추출
- `dataset/prompts/test/mind_uid_dict.pkl` 파일 생성
- `dataset/prompts/test/mind_ui_token.pkl` 파일 생성 (토큰 매핑)

### 2. 사용자/아이템 프로필 파일 생성

다음으로 사용자 프로필과 아이템 프로필을 생성합니다:

```bash
python run_recbole.py \
  --dataset=mind \
  --model=SASRec \
  --config_files=dataset/mind/mind.yaml \
  --gpu_id=1 \
  --dump_profile=True \
  --test_v=test/
```

이 명령어는:
- `dataset/prompts/test/mind_chat.pkl` 파일 생성
- 사용자 프로필, 아이템 프로필, 아이템 ID-이름 매핑 포함

### 3. ICL (In-Context Learning) 예제 파일 생성

`dataset/prompts/test/mind_ICL.json` 파일을 생성해야 합니다. 이 파일은 LLM에게 제공할 예제를 포함합니다.

기본 구조는 다음과 같습니다:
```json
{
  "think_sample": "User profile: ...\nHistory: ...\nTask: Top 10 news articles.\nThought 1: ...\nAction 1: Retrieve[None, 10]\n...",
  "think_only_sample": "...",
  "plan_sample": "...",
  "ranking_sample": "..."
}
```

현재는 `mind_pattern.json`만 생성되었으므로, `mind_ICL.json`을 수동으로 생성하거나 다른 데이터셋의 예제를 참고하여 생성해야 합니다.

## LLM-Empowered Recommendation 실행

프롬프트 파일이 준비되면 `chat_RecAct.py`를 실행할 수 있습니다:

```bash
python chat_RecAct.py \
  --start=0 \
  --step_num=100
```

이 명령어는:
- 첫 100명의 사용자에 대해 LLM-empowered recommendation 실행
- 결과는 `chat_his/mind/start_0.json`에 저장됨

### 주요 설정 확인

`utils.py`에서 다음 설정을 확인하세요:
- `dataset_name="mind"`: MIND 데이터셋 사용
- `test_version="test/"`: 테스트 버전 경로
- `backbone_model="SASRec"`: 기본 모델
- `model_file_dict`: 모델 파일 경로가 올바르게 설정되어 있는지 확인

## 필요한 파일 목록

LLM recommendation 실행을 위해 다음 파일들이 필요합니다:

1. `dataset/prompts/test/mind_uid_dict.pkl` - 테스트 사용자 히스토리
2. `dataset/prompts/test/mind_ui_token.pkl` - 토큰 매핑
3. `dataset/prompts/test/mind_chat.pkl` - 사용자/아이템 프로필
4. `dataset/prompts/mind_pattern.json` - 프롬프트 패턴 (✅ 생성 완료)
5. `dataset/prompts/test/mind_ICL.json` - ICL 예제 (생성 필요)

## 주의사항

- `chat_api.py`에서 LLM API 설정이 올바른지 확인하세요
- GPU 메모리가 충분한지 확인하세요
- 첫 실행 시 작은 `step_num`으로 테스트하는 것을 권장합니다


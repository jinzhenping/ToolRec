
import torch
from recbole.utils.case_study import full_sort_topk, full_sort_scores
import pandas as pd
from recbole.quick_start import load_data_and_model
import pickle
import numpy as np
from recbole.utils import get_trainer
from utils import *

# 모델 캐시: 메모리 누수 방지를 위해 모델을 한 번만 로드하고 재사용
_model_cache = {}

# 속성 값 캐시: 데이터셋별 유효한 속성 값 저장
_attribute_values_cache = {}

def _get_cache_key(dataset, condition, mode):
    """캐시 키 생성"""
    return f"{dataset}_{condition}_{mode}"

def _get_valid_attribute_values(dataset_obj, condition):
    """
    데이터셋에서 유효한 속성 값 목록을 가져옵니다.
    
    Args:
        dataset_obj: RecBole Dataset 객체
        condition: 속성 이름 ('category', 'subcategory' 등)
    
    Returns:
        set: 유효한 속성 값들의 집합 (정규화된 형태)
    """
    cache_key = f"{dataset_obj.dataset_name}_{condition}"
    
    if cache_key not in _attribute_values_cache:
        valid_values = set()
        
        if dataset_obj.item_feat is not None:
            try:
                # item_feat를 pandas DataFrame으로 변환
                item_feat_df = dataset_obj.item_feat
                
                # 타입 확인 및 디버깅
                if not hasattr(item_feat_df, 'columns'):
                    print(f"[디버깅] item_feat 타입: {type(item_feat_df)}")
                    print(f"[디버깅] item_feat 속성: {dir(item_feat_df)[:10]}")
                    # Interaction 객체인 경우 처리
                    if hasattr(item_feat_df, 'interaction'):
                        # Interaction 객체를 pandas DataFrame으로 변환
                        item_dict = {}
                        for key in item_feat_df.interaction.keys():
                            val = item_feat_df.interaction[key]
                            if hasattr(val, 'numpy'):
                                item_dict[key] = val.numpy()
                            elif hasattr(val, 'tolist'):
                                item_dict[key] = val.tolist()
                            elif isinstance(val, torch.Tensor):
                                item_dict[key] = val.cpu().numpy()
                            else:
                                item_dict[key] = val
                        item_feat_df = pd.DataFrame(item_dict)
                
                if hasattr(item_feat_df, 'columns') and condition in item_feat_df.columns:
                    # 모든 유효한 속성 값 추출
                    series = item_feat_df[condition]
                    
                    # series가 Tensor나 numpy array인 경우 pandas Series로 변환
                    if isinstance(series, torch.Tensor):
                        numpy_data = series.cpu().numpy()
                        # 1차원으로 변환
                        if numpy_data.ndim > 1:
                            numpy_data = numpy_data.flatten()
                        series = pd.Series(numpy_data)
                    elif isinstance(series, np.ndarray):
                        # 1차원으로 변환
                        if series.ndim > 1:
                            series = series.flatten()
                        series = pd.Series(series)
                    elif not isinstance(series, pd.Series):
                        # 다른 타입인 경우 pandas Series로 변환 시도
                        try:
                            # list나 iterable인 경우
                            if hasattr(series, '__iter__') and not isinstance(series, str):
                                series = pd.Series(list(series))
                            else:
                                series = pd.Series([series])
                        except Exception as e:
                            print(f"[경고] series를 pandas Series로 변환할 수 없습니다: {type(series)}, 오류: {str(e)}")
                            series = pd.Series([series])
                    
                    # .values 접근 (pandas Series의 속성)
                    # pandas Series의 .values는 속성이므로 직접 접근
                    # 하지만 일부 경우 .values가 메서드로 인식될 수 있으므로 안전하게 처리
                    try:
                        # 먼저 .values 속성에 직접 접근 시도
                        values_attr = getattr(series, 'values', None)
                        if values_attr is None:
                            raise AttributeError("values 속성이 없습니다")
                        
                        # callable인지 확인 (메서드인 경우)
                        if callable(values_attr):
                            print(f"[경고] series.values가 메서드입니다. 대체 방법을 시도합니다.")
                            # Tensor인 경우
                            if isinstance(series, torch.Tensor):
                                values = series.cpu().numpy()
                            elif hasattr(series, 'to_numpy'):
                                values = series.to_numpy()
                            else:
                                values = list(series)
                        else:
                            # 속성인 경우
                            values = values_attr
                            
                            # values가 여전히 callable인지 확인 (이중 체크)
                            if callable(values):
                                print(f"[경고] values가 여전히 callable입니다. 대체 방법을 시도합니다.")
                                # Tensor인 경우
                                if isinstance(series, torch.Tensor):
                                    values = series.cpu().numpy()
                                elif hasattr(series, 'to_numpy'):
                                    values = series.to_numpy()
                                else:
                                    values = list(series)
                    except (AttributeError, TypeError) as e:
                        # .values가 없는 경우 또는 오류 발생 시
                        print(f"[경고] series.values 접근 실패: {str(e)}. 대체 방법을 시도합니다.")
                        # Tensor 객체인 경우
                        if isinstance(series, torch.Tensor):
                            values = series.cpu().numpy()
                        elif hasattr(series, 'to_numpy'):
                            values = series.to_numpy()
                        elif hasattr(series, 'numpy'):
                            values = series.numpy()
                        elif hasattr(series, 'tolist'):
                            values = series.tolist()
                        else:
                            values = list(series)
                    
                    # values가 iterable인지 확인
                    try:
                        iter(values)
                    except TypeError:
                        print(f"[경고] values가 iterable이 아닙니다. list()로 변환합니다.")
                        values = list(series)
                    
                    # 최종 안전 체크: values가 callable이면 오류
                    if callable(values):
                        print(f"[오류] values가 여전히 callable입니다. list()로 변환합니다.")
                        values = list(series)
                    
                    # values가 numpy array나 Tensor인 경우 처리
                    if isinstance(values, (np.ndarray, torch.Tensor)):
                        if isinstance(values, torch.Tensor):
                            values = values.cpu().numpy()
                        values = values.flatten().tolist()
                    
                    # field2id_token을 사용하여 내부 ID를 외부 토큰으로 변환
                    # condition 필드가 field2id_token에 있는 경우 변환
                    use_token_conversion = False
                    if hasattr(dataset_obj, 'field2id_token') and condition in dataset_obj.field2id_token:
                        try:
                            id_token_map = dataset_obj.field2id_token[condition]
                            use_token_conversion = True
                            print(f"[속성 변환] {condition} 필드의 내부 ID를 외부 토큰으로 변환합니다.")
                        except Exception as e:
                            print(f"[경고] field2id_token 접근 실패: {str(e)}")
                    
                    for value in values:
                        # Tensor나 numpy array인 경우 처리
                        if isinstance(value, torch.Tensor):
                            value = value.item() if value.numel() == 1 else int(value.cpu().numpy())
                        elif isinstance(value, np.ndarray):
                            value = value.item() if value.size == 1 else int(value)
                        
                        # 내부 ID를 외부 토큰으로 변환
                        if use_token_conversion:
                            try:
                                # value를 정수로 변환 시도
                                if isinstance(value, (float, str)):
                                    # '0.0' -> 0 변환
                                    int_value = int(float(value))
                                else:
                                    int_value = int(value)
                                
                                # field2id_token 배열에서 토큰 가져오기
                                if 0 <= int_value < len(id_token_map):
                                    value = id_token_map[int_value]
                                else:
                                    # 범위를 벗어난 경우 원래 값 사용
                                    value = str(value)
                            except (ValueError, TypeError, IndexError) as e:
                                # 변환 실패 시 원래 값 사용
                                value = str(value)
                        
                        if isinstance(value, list):
                            # 리스트인 경우 각 항목 추가
                            for item in value:
                                if item is not None and str(item).strip():
                                    valid_values.add(str(item).strip().lower())
                        else:
                            # 단일 값인 경우
                            if value is not None and str(value).strip():
                                valid_values.add(str(value).strip().lower())
            except Exception as e:
                print(f"[경고] 속성 값 추출 중 오류 발생: {str(e)}")
                import traceback
                traceback.print_exc()
        
        _attribute_values_cache[cache_key] = valid_values
        print(f"[속성 값 캐시] {condition} 속성의 유효한 값 {len(valid_values)}개 로드됨")
        if len(valid_values) <= 20:  # 값이 적으면 모두 출력
            print(f"  - 유효한 값: {sorted(list(valid_values))}")
    
    return _attribute_values_cache[cache_key]

def _normalize_attribute_value(value):
    """
    속성 값을 정규화합니다 (대소문자, 공백 처리).
    
    Args:
        value: 원본 속성 값
    
    Returns:
        str: 정규화된 속성 값
    """
    return str(value).strip().lower()

def _find_best_match(attribute_value, valid_values, threshold=0.8):
    """
    유효하지 않은 속성 값에 대해 가장 유사한 유효한 값을 찾습니다.
    
    Args:
        attribute_value: 입력된 속성 값
        valid_values: 유효한 속성 값들의 집합
        threshold: 유사도 임계값 (0-1)
    
    Returns:
        str or None: 가장 유사한 유효한 값, 임계값 미만이면 None
    """
    from difflib import SequenceMatcher
    
    normalized_input = _normalize_attribute_value(attribute_value)
    
    best_match = None
    best_score = 0.0
    
    for valid_value in valid_values:
        # 문자열 유사도 계산
        similarity = SequenceMatcher(None, normalized_input, valid_value).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = valid_value
    
    if best_score >= threshold:
        return best_match
    return None

def retrieval_topk(dataset, condition='None', user_id=None, topK=10, mode='freeze', attribute_value=None):
    """
    Retrieval top K items with optional attribute filtering.
    
    Args:
        dataset: Dataset name
        condition: Attribute type ('None', 'category', 'subcategory')
        user_id: User ID(s)
        topK: Number of items to retrieve
        mode: Model mode ('freeze' or other)
        attribute_value: Specific attribute value to filter by (e.g., 'sports', 'politics')
                        If None, no filtering is applied
    """
    # 캐시 키 생성
    cache_key = _get_cache_key(dataset, condition, mode)
    
    # 캐시에 모델이 없으면 로드
    if cache_key not in _model_cache:
        try:
            model_name = model_file_dict[backbone_model][dataset][condition]
            if mode != 'freeze':
                model_name = model_BERT[backbone_model][dataset][condition]
            model_file = checkpoint_path + model_name
            
            print(f"[메모리 최적화] 모델 로드 중: {model_name}")
            print(f"[메모리 최적화] 모델 파일 경로: {model_file}")
            # load trained model
            config, model, dataset_obj, train_data, valid_data, test_data = load_data_and_model(
                model_file=model_file,
            )
            
            # 모델을 eval 모드로 설정
            model.eval()
            
            # 캐시에 저장
            _model_cache[cache_key] = {
                'config': config,
                'model': model,
                'dataset': dataset_obj,
                'test_data': test_data,
                'train_data': train_data  # train_data도 캐시에 저장
            }
            print(f"[메모리 최적화] 모델 캐시에 저장 완료: {cache_key}")
        except Exception as e:
            print(f"[오류] 모델 로드 실패: {str(e)}")
            print(f"[오류] 캐시 키: {cache_key}")
            print(f"[오류] dataset: {dataset}, condition: {condition}, mode: {mode}")
            import traceback
            traceback.print_exc()
            raise  # 예외를 다시 발생시켜서 호출자에게 알림
    
    # 캐시에서 모델 가져오기
    if cache_key not in _model_cache:
        raise KeyError(f"모델이 캐시에 없습니다: {cache_key}. 모델 로딩이 실패했을 수 있습니다.")
    
    cached = _model_cache[cache_key]
    config = cached['config']
    model = cached['model']
    dataset_obj = cached['dataset']
    test_data = cached['test_data']
    train_data = cached.get('train_data', None)  # train_data 가져오기 (없을 수 있음)
    
    # test_data가 sequential이고 inter_feat가 비어있는 경우, dataset 객체의 inter_feat를 사용
    # sequential 모델의 경우 full_sort_scores가 test_data.dataset.inter_feat를 사용하므로,
    # test_data.dataset.inter_feat를 dataset_obj.inter_feat로 임시 교체
    # full_sort_scores 호출 후 원래대로 복원
    use_dataset_obj_inter_feat = False
    original_inter_feat = None
    if hasattr(test_data, 'is_sequential') and test_data.is_sequential:
        if hasattr(test_data, 'dataset') and hasattr(test_data.dataset, 'inter_feat'):
            if len(test_data.dataset.inter_feat) == 0:
                print(f"[경고] test_data.dataset.inter_feat가 비어있습니다. dataset_obj.inter_feat를 사용합니다.")
                if hasattr(dataset_obj, 'inter_feat') and len(dataset_obj.inter_feat) > 0:
                    # test_data.dataset.inter_feat를 임시로 dataset_obj.inter_feat로 교체
                    try:
                        original_inter_feat = test_data.dataset.inter_feat
                        test_data.dataset.inter_feat = dataset_obj.inter_feat
                        use_dataset_obj_inter_feat = True
                    except (AttributeError, TypeError, ValueError) as e:
                        print(f"[경고] inter_feat 교체 실패: {e}")
                        # 다른 방법 시도: full_sort_scores를 래핑
                        pass
    
    # retrieval top K items, and the corresponding score.
    print(f"[디버깅] user_id 입력: {user_id}, type: {type(user_id)}")
    uid_series = dataset_obj.token2id(dataset_obj.uid_field, user_id)
    print(f"[디버깅] uid_series 결과: {uid_series}, type: {type(uid_series)}")
    
    # uid_series를 list나 numpy array로 변환 (full_sort_scores가 기대하는 형식)
    if isinstance(uid_series, torch.Tensor):
        uid_series = uid_series.cpu().numpy()
    elif not isinstance(uid_series, (list, np.ndarray)):
        uid_series = np.array([uid_series])
    
    # 1차원 배열로 변환
    if isinstance(uid_series, np.ndarray) and uid_series.ndim == 0:
        uid_series = np.array([uid_series])
    elif isinstance(uid_series, np.ndarray) and uid_series.ndim > 1:
        uid_series = uid_series.flatten()
    
    print(f"[디버깅] uid_series 변환 후: {uid_series}, type: {type(uid_series)}, shape: {uid_series.shape if hasattr(uid_series, 'shape') else len(uid_series)}")
    
    if len(uid_series) == 0:
        print(f"[경고] uid_series가 비어있습니다! user_id를 확인하세요: {user_id}")
        batch_size = 1  # 기본값
        return (
            torch.zeros((batch_size, 0), device=config["device"]),
            [[] for _ in range(batch_size)],
            np.array([[] for _ in range(batch_size)])
        )

    # 속성 값 필터링이 필요한 경우, 전체 아이템 점수를 가져온 후 필터링
    if attribute_value and condition != 'None' and dataset_obj.item_feat is not None:
        # 유효한 속성 값 목록 가져오기
        valid_values = _get_valid_attribute_values(dataset_obj, condition)
        normalized_input = _normalize_attribute_value(attribute_value)
        
        # 속성 값 유효성 검증
        if normalized_input not in valid_values:
            # 유효하지 않은 값인 경우, 가장 유사한 값 찾기
            best_match = _find_best_match(attribute_value, valid_values, threshold=0.7)
            
            if best_match:
                print(f"[경고] '{attribute_value}'는 유효하지 않은 {condition} 값입니다.")
                print(f"      가장 유사한 유효한 값 '{best_match}'를 사용합니다.")
                attribute_value = best_match  # 유사한 값으로 대체
            else:
                print(f"[오류] '{attribute_value}'는 유효하지 않은 {condition} 값입니다.")
                print(f"      유효한 값 목록 (일부): {sorted(list(valid_values))[:10]}")
                # 빈 결과 반환
                # uid_series가 list나 numpy array일 수 있음
                if isinstance(uid_series, torch.Tensor):
                    batch_size = uid_series.shape[0]
                elif isinstance(uid_series, (list, np.ndarray)):
                    batch_size = len(uid_series)
                else:
                    batch_size = 1
                return (
                    torch.zeros((batch_size, 0), device=config["device"]),
                    [[] for _ in range(batch_size)],
                    np.array([[] for _ in range(batch_size)])
                )
        
        # 전체 아이템에 대한 점수 계산
        print(f"[디버깅] full_sort_scores 호출 전: uid_series={uid_series}, type={type(uid_series)}")
        print(f"[디버깅] test_data.is_sequential: {test_data.is_sequential if hasattr(test_data, 'is_sequential') else 'N/A'}")
        
        # test_data가 sequential인 경우, dataset.inter_feat에 해당 사용자가 있는지 확인
        # 하지만 full_sort_scores를 먼저 시도하고, 실패하면 대체 방법 사용
        if hasattr(test_data, 'is_sequential') and test_data.is_sequential:
            if hasattr(test_data, 'dataset') and hasattr(test_data.dataset, 'inter_feat'):
                uid_list = list(uid_series) if isinstance(uid_series, (list, np.ndarray)) else [uid_series]
                uid_field = test_data.dataset.uid_field
                inter_feat_uids = test_data.dataset.inter_feat[uid_field].unique().numpy() if len(test_data.dataset.inter_feat) > 0 else np.array([])
                print(f"[디버깅] dataset.inter_feat에 있는 사용자 ID 수: {len(inter_feat_uids)}")
                print(f"[디버깅] 요청한 사용자 ID: {uid_list}")
                
                # test_data.dataset.inter_feat가 비어있거나 사용자가 없으면 dataset_obj.inter_feat 사용
                if len(inter_feat_uids) == 0 or any(int(uid) not in inter_feat_uids for uid in uid_list):
                    print(f"[경고] test_data.dataset.inter_feat에 사용자가 없습니다. dataset_obj.inter_feat를 사용합니다.")
                    # dataset 객체의 inter_feat를 직접 사용
                    if hasattr(dataset_obj, 'inter_feat') and len(dataset_obj.inter_feat) > 0:
                        # dataset_obj.inter_feat에 사용자가 있는지 확인
                        dataset_obj_inter_feat_uids = dataset_obj.inter_feat[uid_field].unique().numpy()
                        print(f"[디버깅] dataset_obj.inter_feat에 있는 사용자 ID 수: {len(dataset_obj_inter_feat_uids)}")
                        print(f"[디버깅] dataset_obj.inter_feat의 사용자 ID 범위: {dataset_obj_inter_feat_uids.min()} ~ {dataset_obj_inter_feat_uids.max()}" if len(dataset_obj_inter_feat_uids) > 0 else "N/A")
                        
                        # dataset_obj.inter_feat에 사용자가 있으면 교체
                        if len(dataset_obj_inter_feat_uids) > 0 and any(int(uid) in dataset_obj_inter_feat_uids for uid in uid_list):
                            # test_data.dataset.inter_feat를 임시로 dataset_obj.inter_feat로 교체
                            if not use_dataset_obj_inter_feat:
                                original_inter_feat = test_data.dataset.inter_feat
                                test_data.dataset.inter_feat = dataset_obj.inter_feat
                                use_dataset_obj_inter_feat = True
                                print(f"[디버깅] test_data.dataset.inter_feat를 dataset_obj.inter_feat로 임시 교체")
                        else:
                            print(f"[경고] dataset_obj.inter_feat에도 사용자가 없습니다!")
                            # 빈 결과 반환
                            batch_size = len(uid_series) if hasattr(uid_series, '__len__') else 1
                            return (
                                torch.zeros((batch_size, 0), device=config["device"]),
                                [[] for _ in range(batch_size)],
                                np.array([[] for _ in range(batch_size)])
                            )
                    else:
                        print(f"[경고] dataset.inter_feat도 없거나 비어있습니다!")
                        # 빈 결과 반환
                        batch_size = len(uid_series) if hasattr(uid_series, '__len__') else 1
                        return (
                            torch.zeros((batch_size, 0), device=config["device"]),
                            [[] for _ in range(batch_size)],
                            np.array([[] for _ in range(batch_size)])
                        )
        
        # test_data가 sequential이 아닌 경우, uid2history_item에 사용자가 있는지 확인
        if hasattr(test_data, 'is_sequential') and not test_data.is_sequential:
            if hasattr(test_data, 'uid2history_item'):
                uid_list = list(uid_series) if isinstance(uid_series, (list, np.ndarray)) else [uid_series]
                # uid2history_item은 numpy array이므로 인덱스로 접근
                # 사용자 ID가 배열 크기 내에 있는지 확인
                user_num = len(test_data.uid2history_item)
                invalid_uids = [uid for uid in uid_list if int(uid) >= user_num or int(uid) < 0]
                if invalid_uids:
                    print(f"[경고] 다음 사용자 ID가 유효 범위를 벗어났습니다: {invalid_uids}")
                    print(f"[경고] 사용 가능한 사용자 ID 범위: 0 ~ {user_num-1}")
                    print(f"[경고] dataset user_num: {dataset_obj.user_num}")
                    # 빈 결과 반환
                    batch_size = len(uid_series) if hasattr(uid_series, '__len__') else 1
                    return (
                        torch.zeros((batch_size, 0), device=config["device"]),
                        [[] for _ in range(batch_size)],
                        np.array([[] for _ in range(batch_size)])
                    )
                # 유효한 사용자인지 확인 (uid2history_item[uid]가 None이 아닌지)
                missing_uids = []
                for uid in uid_list:
                    uid_int = int(uid)
                    if uid_int < user_num and (test_data.uid2history_item[uid_int] is None or len(test_data.uid2history_item[uid_int]) == 0):
                        missing_uids.append(uid)
                if missing_uids:
                    print(f"[경고] test_data.uid2history_item에 다음 사용자의 히스토리가 없습니다: {missing_uids}")
                    print(f"[경고] 사용 가능한 사용자 수: {user_num}")
                    # 빈 결과 반환
                    batch_size = len(uid_series) if hasattr(uid_series, '__len__') else 1
                    return (
                        torch.zeros((batch_size, 0), device=config["device"]),
                        [[] for _ in range(batch_size)],
                        np.array([[] for _ in range(batch_size)])
                    )
        
        # full_sort_scores 호출
        # sequential 모델의 경우, dataset.inter_feat 대신 train_data에서 히스토리를 가져와서 사용
        try:
            # sequential 모델이고 test_data.dataset.inter_feat에 사용자가 없는 경우
            # train_data에서 히스토리를 가져와서 직접 input_interaction 생성
            if (hasattr(test_data, 'is_sequential') and test_data.is_sequential and 
                hasattr(test_data, 'dataset') and hasattr(test_data.dataset, 'inter_feat')):
                uid_list = list(uid_series) if isinstance(uid_series, (list, np.ndarray)) else [uid_series]
                uid_field = test_data.dataset.uid_field
                inter_feat_uids = test_data.dataset.inter_feat[uid_field].unique().numpy() if len(test_data.dataset.inter_feat) > 0 else np.array([])
                
                # test_data.dataset.inter_feat에 사용자가 없으면 train_data에서 히스토리 가져오기
                if len(inter_feat_uids) == 0 or any(int(uid) not in inter_feat_uids for uid in uid_list):
                    print(f"[경고] test_data.dataset.inter_feat에 사용자가 없습니다. train_data에서 히스토리를 가져와 사용합니다.")
                    # train_data에서 사용자 히스토리 가져오기
                    from recbole.data.interaction import Interaction
                    
                    # train_data가 캐시에 있는지 확인
                    if train_data is not None:
                        # train_data에서 사용자 히스토리 찾기
                        train_inter_feat = train_data.dataset.inter_feat if hasattr(train_data, 'dataset') else None
                        if train_inter_feat is not None and len(train_inter_feat) > 0:
                            # 사용자의 마지막 interaction 가져오기 (sequential 모델은 마지막 시퀀스를 사용)
                            input_interactions = []
                            for uid in uid_list:
                                uid_tensor = torch.tensor([int(uid)], dtype=torch.long)
                                # train_inter_feat에서 해당 사용자의 interaction 찾기
                                user_mask = train_inter_feat[uid_field] == int(uid)
                                user_interactions = train_inter_feat[user_mask]
                                
                                if len(user_interactions) > 0:
                                    # 마지막 interaction 사용 (가장 최근 히스토리)
                                    last_interaction = user_interactions[-1]
                                    input_interactions.append(last_interaction)
                                else:
                                    # 사용자가 train_data에도 없으면 빈 interaction 생성
                                    print(f"[경고] 사용자 {uid}가 train_data에도 없습니다.")
                                    # 빈 interaction은 나중에 처리
                                    input_interactions.append(None)
                            
                            # input_interaction 생성
                            if all(inter is not None for inter in input_interactions):
                                # 모든 interaction을 하나로 합치기
                                combined_interaction = {}
                                for key in input_interactions[0].interaction.keys():
                                    combined_interaction[key] = torch.stack([inter.interaction[key] for inter in input_interactions])
                                
                                input_interaction = Interaction(combined_interaction)
                                
                                # 모델로 점수 계산
                                device = config["device"]
                                input_interaction = input_interaction.to(device)
                                try:
                                    scores = model.full_sort_predict(input_interaction)
                                except NotImplementedError:
                                    input_interaction = input_interaction.repeat_interleave(dataset_obj.item_num)
                                    input_interaction.update(
                                        test_data.dataset.get_item_feature().to(device).repeat(len(uid_series))
                                    )
                                    scores = model.predict(input_interaction)
                                
                                all_scores = scores.view(-1, dataset_obj.item_num)
                                all_scores[:, 0] = -np.inf  # set scores of [pad] to -inf
                                print(f"[디버깅] train_data에서 히스토리를 가져와 계산한 all_scores shape: {all_scores.shape}")
                            else:
                                # 일부 사용자가 train_data에 없으면 일반 full_sort_scores 사용
                                print(f"[경고] 일부 사용자가 train_data에 없습니다. 일반 full_sort_scores를 사용합니다.")
                                all_scores = full_sort_scores(
                                    uid_series, model, test_data, device=config["device"]
                                )
                        else:
                            # train_data가 없으면 일반 full_sort_scores 사용
                            print(f"[경고] train_data를 사용할 수 없습니다. 일반 full_sort_scores를 사용합니다.")
                            all_scores = full_sort_scores(
                                uid_series, model, test_data, device=config["device"]
                            )
                    else:
                        # train_data가 없으면 일반 full_sort_scores 사용
                        all_scores = full_sort_scores(
                            uid_series, model, test_data, device=config["device"]
                        )
                else:
                    # test_data.dataset.inter_feat에 사용자가 있으면 일반 full_sort_scores 사용
                    all_scores = full_sort_scores(
                        uid_series, model, test_data, device=config["device"]
                    )
            else:
                # sequential 모델이 아니면 일반 full_sort_scores 사용
                all_scores = full_sort_scores(
                    uid_series, model, test_data, device=config["device"]
                )
            
            print(f"[디버깅] all_scores shape: {all_scores.shape if isinstance(all_scores, torch.Tensor) else type(all_scores)}")
            
            # all_scores가 비어있으면 dataset_obj.inter_feat를 사용하여 재시도
            if isinstance(all_scores, torch.Tensor) and all_scores.shape[0] == 0:
                print(f"[경고] all_scores가 빈 텐서입니다. dataset_obj.inter_feat를 사용하여 재시도합니다.")
                if hasattr(dataset_obj, 'inter_feat') and len(dataset_obj.inter_feat) > 0:
                    # test_data.dataset.inter_feat를 임시로 dataset_obj.inter_feat로 교체
                    if not use_dataset_obj_inter_feat:
                        original_inter_feat = test_data.dataset.inter_feat
                        test_data.dataset.inter_feat = dataset_obj.inter_feat
                        use_dataset_obj_inter_feat = True
                        print(f"[디버깅] test_data.dataset.inter_feat를 dataset_obj.inter_feat로 임시 교체 (재시도)")
                    
                    # 재시도
                    all_scores = full_sort_scores(
                        uid_series, model, test_data, device=config["device"]
                    )
                    print(f"[디버깅] 재시도 후 all_scores shape: {all_scores.shape if isinstance(all_scores, torch.Tensor) else type(all_scores)}")
        except Exception as e:
            print(f"[경고] full_sort_scores 호출 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            # 빈 결과 반환
            batch_size = len(uid_series) if hasattr(uid_series, '__len__') else 1
            return (
                torch.zeros((batch_size, 0), device=config["device"]),
                [[] for _ in range(batch_size)],
                np.array([[] for _ in range(batch_size)])
            )
        
        # inter_feat를 원래대로 복원
        if use_dataset_obj_inter_feat and original_inter_feat is not None:
            try:
                test_data.dataset.inter_feat = original_inter_feat
            except (AttributeError, TypeError, ValueError):
                pass
        
        # all_scores의 batch_size가 0인 경우 확인
        if isinstance(all_scores, torch.Tensor) and all_scores.shape[0] == 0:
            print(f"[경고] all_scores의 batch_size가 0입니다! uid_series를 확인하세요: {uid_series}")
            print(f"[경고] test_data.is_sequential: {test_data.is_sequential if hasattr(test_data, 'is_sequential') else 'N/A'}")
            # 빈 결과 반환
            batch_size = len(uid_series) if hasattr(uid_series, '__len__') else 1
            return (
                torch.zeros((batch_size, 0), device=config["device"]),
                [[] for _ in range(batch_size)],
                np.array([[] for _ in range(batch_size)])
            )
        
        # all_scores가 Tensor가 아닌 경우 처리
        if not isinstance(all_scores, torch.Tensor):
            if isinstance(all_scores, (list, np.ndarray)):
                all_scores = torch.tensor(all_scores, device=config["device"])
            else:
                print(f"[경고] all_scores 타입이 예상과 다릅니다: {type(all_scores)}")
                # 일반 검색으로 대체
                topk_score, topk_iid_list = full_sort_topk(
                    uid_series, model, test_data, k=topK, device=config["device"]
                )
                external_item_list = dataset_obj.id2token(dataset_obj.iid_field, topk_iid_list.cpu())
                external_item_list_name = []
                for u_list in external_item_list:
                    external_item_list_name.append([itemID_name.get(iid, '') for iid in u_list])
                external_item_list_name = np.array(external_item_list_name)
                return topk_score, external_item_list, external_item_list_name
        
        # 아이템 속성 정보 가져오기
        item_feat = dataset_obj.item_feat
        
        # item_feat를 pandas DataFrame으로 변환
        if hasattr(item_feat, 'to_pandas'):
            # Interaction 객체인 경우
            item_feat = item_feat.to_pandas()
        elif not hasattr(item_feat, 'columns'):
            # pandas DataFrame이 아닌 경우
            print(f"[경고] item_feat가 pandas DataFrame이 아닙니다. 일반 검색을 수행합니다.")
            topk_score, topk_iid_list = full_sort_topk(
                uid_series, model, test_data, k=topK, device=config["device"]
            )
            external_item_list = dataset_obj.id2token(dataset_obj.iid_field, topk_iid_list.cpu())
            external_item_list_name = []
            for u_list in external_item_list:
                external_item_list_name.append([itemID_name.get(iid, '') for iid in u_list])
            external_item_list_name = np.array(external_item_list_name)
            return topk_score, external_item_list, external_item_list_name
        
        if condition in item_feat.columns:
            # 속성 값으로 필터링할 아이템 인덱스 찾기
            # MIND 데이터셋의 category/subcategory는 리스트나 문자열일 수 있음
            
            # field2token_id를 사용하여 외부 토큰을 내부 ID로 변환
            # LLM이 입력한 attribute_value(예: 'sports')를 내부 ID로 변환
            target_internal_id = None
            if hasattr(dataset_obj, 'field2token_id') and condition in dataset_obj.field2token_id:
                try:
                    token_id_map = dataset_obj.field2token_id[condition]
                    normalized_input = _normalize_attribute_value(attribute_value)
                    # 정규화된 입력으로 검색
                    for token, internal_id in token_id_map.items():
                        if _normalize_attribute_value(token) == normalized_input:
                            target_internal_id = internal_id
                            print(f"[속성 변환] '{attribute_value}' -> 내부 ID {target_internal_id}")
                            break
                except Exception as e:
                    print(f"[경고] field2token_id 변환 실패: {str(e)}")
            
            # item_feat[condition]이 Tensor나 numpy array인 경우 pandas Series로 변환
            condition_series = item_feat[condition]
            if isinstance(condition_series, torch.Tensor):
                numpy_data = condition_series.cpu().numpy()
                # 1차원으로 변환
                if numpy_data.ndim > 1:
                    numpy_data = numpy_data.flatten()
                condition_series = pd.Series(numpy_data)
            elif isinstance(condition_series, np.ndarray):
                # 1차원으로 변환
                if condition_series.ndim > 1:
                    condition_series = condition_series.flatten()
                condition_series = pd.Series(condition_series)
            elif not isinstance(condition_series, pd.Series):
                # 다른 타입인 경우 pandas Series로 변환 시도
                try:
                    # list나 iterable인 경우
                    if hasattr(condition_series, '__iter__') and not isinstance(condition_series, str):
                        condition_list = list(condition_series)
                        condition_series = pd.Series(condition_list)
                    else:
                        condition_series = pd.Series([condition_series])
                except Exception as e:
                    print(f"[경고] condition_series를 pandas Series로 변환 실패: {str(e)}")
                    # 일반 검색으로 대체
                    topk_score, topk_iid_list = full_sort_topk(
                        uid_series, model, test_data, k=topK, device=config["device"]
                    )
                    external_item_list = dataset_obj.id2token(dataset_obj.iid_field, topk_iid_list.cpu())
                    external_item_list_name = []
                    for u_list in external_item_list:
                        external_item_list_name.append([itemID_name.get(iid, '') for iid in u_list])
                    external_item_list_name = np.array(external_item_list_name)
                    return topk_score, external_item_list, external_item_list_name
            
            def matches_attribute(row_value, target_value, target_id=None):
                """속성 값 매칭 (리스트, 문자열 모두 지원)"""
                normalized_target = _normalize_attribute_value(target_value)
                
                # row_value가 숫자인 경우 (내부 ID) 처리
                if target_id is not None:
                    try:
                        # row_value를 정수로 변환 시도
                        if isinstance(row_value, (float, str)):
                            row_int = int(float(row_value))
                        else:
                            row_int = int(row_value)
                        
                        # 내부 ID로 직접 비교
                        if row_int == target_id:
                            return True
                    except (ValueError, TypeError):
                        pass
                
                # 외부 토큰으로 비교
                if isinstance(row_value, list):
                    # 리스트인 경우: 리스트 내에 값이 포함되어 있는지 확인
                    return any(_normalize_attribute_value(str(item)) == normalized_target for item in row_value)
                else:
                    # 문자열인 경우: 직접 비교 (대소문자 무시)
                    return _normalize_attribute_value(str(row_value)) == normalized_target
            
            # 필터링 적용
            mask = condition_series.apply(lambda x: matches_attribute(x, attribute_value, target_internal_id))
            filtered_items = item_feat[mask]
            
            if len(filtered_items) > 0:
                # 필터링된 아이템의 외부 ID 가져오기
                try:
                    # pandas Series의 .values 속성 사용
                    iid_series = filtered_items[dataset_obj.iid_field]
                    
                    # iid_series가 Tensor인 경우 처리
                    if isinstance(iid_series, torch.Tensor):
                        filtered_iids = iid_series.cpu().numpy().tolist()
                    elif isinstance(iid_series, np.ndarray):
                        filtered_iids = iid_series.tolist()
                    else:
                        # pandas Series인 경우
                        try:
                            iid_values = iid_series.values
                            # values가 callable인 경우 (함수 객체인 경우) 처리
                            if callable(iid_values):
                                print(f"[경고] iid_series.values가 메서드입니다. 대체 방법을 시도합니다.")
                                if hasattr(iid_series, 'to_numpy'):
                                    filtered_iids = iid_series.to_numpy().tolist()
                                else:
                                    filtered_iids = list(iid_series)
                            else:
                                filtered_iids = iid_values.tolist()
                        except (AttributeError, TypeError) as e:
                            # .values가 없는 경우 또는 오류 발생 시
                            print(f"[경고] iid_series.values 접근 실패: {str(e)}. 대체 방법을 시도합니다.")
                            if hasattr(iid_series, 'to_numpy'):
                                filtered_iids = iid_series.to_numpy().tolist()
                            else:
                                filtered_iids = list(iid_series)
                except Exception as e:
                    # 모든 방법 실패 시
                    print(f"[경고] iid_series 추출 실패: {str(e)}. 직접 접근을 시도합니다.")
                    filtered_iids = list(filtered_items[dataset_obj.iid_field])
                
                # filtered_iids를 문자열로 변환 (token2id는 문자열을 기대)
                filtered_iids_str = [str(iid) for iid in filtered_iids]
                
                # 외부 ID를 내부 ID로 변환
                try:
                    filtered_iid_internal = dataset_obj.token2id(dataset_obj.iid_field, filtered_iids_str)
                except Exception as e:
                    print(f"[경고] token2id 변환 실패: {str(e)}. filtered_iids가 이미 내부 ID일 수 있습니다.")
                    # filtered_iids가 이미 내부 ID인 경우 (정수 리스트)
                    try:
                        filtered_iid_internal = np.array([int(iid) for iid in filtered_iids])
                    except (ValueError, TypeError):
                        print(f"[오류] filtered_iids를 내부 ID로 변환할 수 없습니다: {filtered_iids[:5]}")
                        # 일반 검색으로 대체
                        topk_score, topk_iid_list = full_sort_topk(
                            uid_series, model, test_data, k=topK, device=config["device"]
                        )
                        external_item_list = dataset_obj.id2token(dataset_obj.iid_field, topk_iid_list.cpu())
                        external_item_list_name = []
                        for u_list in external_item_list:
                            external_item_list_name.append([itemID_name.get(iid, '') for iid in u_list])
                        external_item_list_name = np.array(external_item_list_name)
                        return topk_score, external_item_list, external_item_list_name
                
                if len(filtered_iid_internal) > 0:
                    # 필터링된 아이템에 대한 점수만 선택
                    # all_scores의 인덱스는 내부 ID와 일치해야 함
                    # filtered_iid_internal을 텐서로 변환
                    filtered_iid_tensor = torch.tensor(filtered_iid_internal, device=config["device"], dtype=torch.long)
                    filtered_scores = all_scores[:, filtered_iid_tensor]  # shape: [batch_size, num_filtered]
                    
                    print(f"[디버깅] filtered_scores shape: {filtered_scores.shape}")
                    print(f"[디버깅] all_scores shape: {all_scores.shape}")
                    print(f"[디버깅] filtered_iid_tensor shape: {filtered_iid_tensor.shape}")
                    
                    # TopK 선택
                    k = min(topK, len(filtered_iid_internal))
                    topk_scores_filtered, topk_indices_filtered = torch.topk(filtered_scores, k=k, dim=1)
                    
                    print(f"[디버깅] topk_indices_filtered shape: {topk_indices_filtered.shape}")
                    print(f"[디버깅] topk_scores_filtered shape: {topk_scores_filtered.shape}")
                    
                    # 필터링된 인덱스를 원래 아이템 인덱스로 변환
                    # topk_indices_filtered는 [batch_size, k] 형태
                    # filtered_iid_tensor는 [num_filtered] 형태
                    # topk_iid_list는 [batch_size, k] 형태가 되어야 함
                    batch_size = topk_indices_filtered.shape[0]
                    topk_iid_list = torch.zeros((batch_size, k), dtype=torch.long, device=config["device"])
                    for b in range(batch_size):
                        topk_iid_list[b] = filtered_iid_tensor[topk_indices_filtered[b]]
                    topk_score = topk_scores_filtered
                    
                    print(f"[필터링] {condition}='{attribute_value}' 조건으로 {len(filtered_iids)}개 아이템 중 {k}개 선택")
                    print(f"[디버깅] topk_iid_list shape: {topk_iid_list.shape}, topk_score shape: {topk_score.shape}")
                    if topk_iid_list.numel() > 0:
                        print(f"[디버깅] topk_iid_list 처음 5개: {topk_iid_list[0][:5] if topk_iid_list.dim() > 1 else topk_iid_list[:5]}")
                        print(f"[디버깅] topk_score 처음 5개: {topk_score[0][:5] if topk_score.dim() > 1 else topk_score[:5]}")
                else:
                    # 내부 ID 변환 실패
                    print(f"[경고] {condition}='{attribute_value}' 조건에 맞는 아이템의 내부 ID 변환 실패")
                    batch_size = uid_series.shape[0]
                    topk_score = torch.zeros((batch_size, 0), device=config["device"])
                    topk_iid_list = torch.zeros((batch_size, 0), dtype=torch.long, device=config["device"])
            else:
                # 필터링된 아이템이 없으면 빈 결과 반환
                print(f"[경고] {condition}='{attribute_value}' 조건에 맞는 아이템이 없습니다.")
                batch_size = uid_series.shape[0]
                topk_score = torch.zeros((batch_size, 0), device=config["device"])
                topk_iid_list = torch.zeros((batch_size, 0), dtype=torch.long, device=config["device"])
        else:
            # 속성이 없으면 일반 검색 수행
            print(f"[경고] {condition} 속성이 item_feat에 없습니다. 일반 검색을 수행합니다.")
            topk_score, topk_iid_list = full_sort_topk(
                uid_series, model, test_data, k=topK, device=config["device"]
            )
    else:
        # 필터링이 필요 없으면 일반 검색 수행
        topk_score, topk_iid_list = full_sort_topk(
            uid_series, model, test_data, k=topK, device=config["device"]
        )
    
    # print(topk_score)  # scores of top 10 items
    # print(topk_iid_list)  # internal id of top 10 items
    
    # 디버깅: topk_iid_list 확인
    if topk_iid_list.numel() == 0:
        print(f"[경고] topk_iid_list가 비어있습니다. shape: {topk_iid_list.shape}")
        print(f"[경고] topk_score shape: {topk_score.shape}")
        # 빈 결과 반환
        batch_size = uid_series.shape[0] if isinstance(uid_series, torch.Tensor) else len(uid_series)
        return (
            torch.zeros((batch_size, 0), device=config["device"]),
            [[] for _ in range(batch_size)],
            np.array([[] for _ in range(batch_size)])
        )
    
    print(f"[디버깅] topk_iid_list shape: {topk_iid_list.shape}, topk_score shape: {topk_score.shape}")
    if topk_iid_list.numel() > 0:
        print(f"[디버깅] topk_iid_list 처음 5개: {topk_iid_list[0][:5] if topk_iid_list.dim() > 1 else topk_iid_list[:5]}")
        print(f"[디버깅] topk_score 처음 5개: {topk_score[0][:5] if topk_score.dim() > 1 else topk_score[:5]}")
    
    external_item_list = dataset_obj.id2token(dataset_obj.iid_field, topk_iid_list.cpu())
    print(f"[디버깅] external_item_list type: {type(external_item_list)}, length: {len(external_item_list) if hasattr(external_item_list, '__len__') else 'N/A'}")
    if len(external_item_list) > 0:
        print(f"[디버깅] external_item_list[0] type: {type(external_item_list[0])}, length: {len(external_item_list[0]) if hasattr(external_item_list[0], '__len__') else 'N/A'}")
        if len(external_item_list[0]) > 0:
            print(f"[디버깅] external_item_list[0] 처음 5개: {external_item_list[0][:5]}")
    # print(external_item_list)
    external_item_list_name = []
    for u_list in external_item_list:
        external_item_list_name.append([itemID_name.get(iid, '') for iid in u_list])
    external_item_list_name = np.array(external_item_list_name)


    return topk_score, external_item_list, external_item_list_name

def get_cached_model(dataset, condition='None', mode='freeze'):
    """
    캐시에서 모델을 가져옵니다. 캐시에 없으면 로드합니다.
    
    Args:
        dataset: Dataset name
        condition: Attribute type ('None', 'category', 'subcategory')
        mode: Model mode ('freeze' or other)
    
    Returns:
        tuple: (config, model, dataset_obj, test_data) 또는 None (로드 실패 시)
    """
    global _model_cache
    cache_key = _get_cache_key(dataset, condition, mode)
    
    # 캐시에 모델이 없으면 로드
    if cache_key not in _model_cache:
        try:
            model_name = model_file_dict[backbone_model][dataset][condition]
            if mode != 'freeze':
                model_name = model_BERT[backbone_model][dataset][condition]
            model_file = checkpoint_path + model_name
            
            print(f"[메모리 최적화] 모델 로드 중 (캐시): {model_name}")
            print(f"[메모리 최적화] 모델 파일 경로: {model_file}")
            # load trained model
            config, model, dataset_obj, train_data, valid_data, test_data = load_data_and_model(
                model_file=model_file,
            )
            
            # 모델을 eval 모드로 설정
            model.eval()
            
            # 캐시에 저장
            _model_cache[cache_key] = {
                'config': config,
                'model': model,
                'dataset': dataset_obj,
                'test_data': test_data
            }
            print(f"[메모리 최적화] 모델 캐시에 저장 완료: {cache_key}")
        except Exception as e:
            print(f"[오류] 모델 로드 실패 (캐시): {str(e)}")
            print(f"[오류] 캐시 키: {cache_key}")
            print(f"[오류] dataset: {dataset}, condition: {condition}, mode: {mode}")
            import traceback
            traceback.print_exc()
            return None
    
    # 캐시에서 모델 가져오기
    if cache_key not in _model_cache:
        return None
    
    cached = _model_cache[cache_key]
    return cached['config'], cached['model'], cached['dataset'], cached['test_data']

def clear_model_cache():
    """모델 캐시 정리 (메모리 해제)"""
    global _model_cache
    import gc
    for key in list(_model_cache.keys()):
        del _model_cache[key]
    _model_cache = {}
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("[메모리 최적화] 모델 캐시 정리 완료")

def stdout_retrived_items(score, item_id, item_name):
    retrived_items = []
    
    # score가 Tensor인 경우 numpy array로 변환
    if isinstance(score, torch.Tensor):
        score = score.cpu().numpy()
    
    # item_id가 list가 아닌 경우 처리
    if not isinstance(item_id, list):
        if isinstance(item_id, np.ndarray):
            item_id = item_id.tolist()
        elif isinstance(item_id, torch.Tensor):
            item_id = item_id.cpu().tolist()
        else:
            item_id = [item_id]
    
    # item_name이 list가 아닌 경우 처리
    if not isinstance(item_name, (list, np.ndarray)):
        if isinstance(item_name, torch.Tensor):
            item_name = item_name.cpu().tolist()
        else:
            item_name = [item_name]
    elif isinstance(item_name, np.ndarray):
        item_name = item_name.tolist()
    
    # batch_size 확인 (item_id의 길이 사용)
    batch_size = len(item_id)
    
    for n in range(batch_size):
        item_strings = ""
        # score, item_id, item_name의 n번째 요소 가져오기
        if isinstance(score, (list, np.ndarray)):
            score_n = score[n]
        else:
            score_n = score[n] if hasattr(score, '__getitem__') else [score]
        
        item_id_n = item_id[n] if n < len(item_id) else []
        item_name_n = item_name[n] if n < len(item_name) else []
        
        # score_n이 단일 값인 경우 리스트로 변환
        if not isinstance(score_n, (list, np.ndarray, torch.Tensor)):
            score_n = [score_n]
        elif isinstance(score_n, torch.Tensor):
            score_n = score_n.cpu().numpy()
        
        # item_id_n과 item_name_n도 리스트로 변환
        if not isinstance(item_id_n, list):
            item_id_n = [item_id_n]
        if not isinstance(item_name_n, list):
            item_name_n = [item_name_n]
        
        # 길이 맞추기 (최소 길이 사용)
        min_len = min(len(score_n), len(item_id_n), len(item_name_n))
        
        # zip으로 순회
        for i in range(min_len):
            s = score_n[i]
            iid = item_id_n[i]
            ina = item_name_n[i]
            
            # s가 Tensor나 numpy array인 경우 float로 변환
            if isinstance(s, torch.Tensor):
                s_val = s.item() if s.numel() == 1 else float(s.cpu().numpy())
            elif isinstance(s, np.ndarray):
                s_val = s.item() if s.size == 1 else float(s)
            else:
                s_val = float(s)
            
            item_strings = item_strings + str(iid) + ', ' + str(ina) + ", " + str(round(s_val, 4)) + "\n"
        retrived_items.append(item_strings)
    return retrived_items

    
# if __name__ == "__main__":
    
#     # test

#     # score = full_sort_scores(uid_series, model, test_data, device=config["device"])
#     # print(score)  # score of all items
#     # print(
#     #     score[0, dataset.token2id(dataset.iid_field, ["242", "302"])]
#     # )  # score of item ['242', '302'] for user '196'.
#     users = ["8", "88", "588", "688", "888"]
#     topK = 6
#     topk_score, external_item_list, external_item_list_name = retrieval_topk(condition='ne', user_id=users, topK=topK)
#     retrived_items = stdout_retrived_items(topk_score, external_item_list, external_item_list_name)

#     topk_score1, external_item_list1, external_item_list_name1 = retrieval_topk(condition='None', user_id=users, topK=topK)
#     retrived_items1 = stdout_retrived_items(topk_score1, external_item_list1, external_item_list_name1)

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
                        series = pd.Series(series.cpu().numpy())
                    elif isinstance(series, np.ndarray):
                        series = pd.Series(series)
                    elif not isinstance(series, pd.Series):
                        # 다른 타입인 경우 pandas Series로 변환 시도
                        try:
                            series = pd.Series(series)
                        except Exception:
                            print(f"[경고] series를 pandas Series로 변환할 수 없습니다: {type(series)}")
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
                    
                    for value in values:
                        # Tensor나 numpy array인 경우 처리
                        if isinstance(value, torch.Tensor):
                            value = value.item() if value.numel() == 1 else str(value.cpu().numpy())
                        elif isinstance(value, np.ndarray):
                            value = value.item() if value.size == 1 else str(value)
                        
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
        model_name = model_file_dict[backbone_model][dataset][condition]
        if mode != 'freeze':
            model_name = model_BERT[backbone_model][dataset][condition]
        model_file = checkpoint_path + model_name
        
        print(f"[메모리 최적화] 모델 로드 중: {model_name}")
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
    
    # 캐시에서 모델 가져오기
    cached = _model_cache[cache_key]
    config = cached['config']
    model = cached['model']
    dataset_obj = cached['dataset']
    test_data = cached['test_data']
    
    # retrieval top K items, and the corresponding score.
    uid_series = dataset_obj.token2id(dataset_obj.uid_field, user_id)

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
        all_scores = full_sort_scores(
            uid_series, model, test_data, device=config["device"]
        )  # shape: [batch_size, num_items]
        
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
            def matches_attribute(row_value, target_value):
                """속성 값 매칭 (리스트, 문자열 모두 지원)"""
                normalized_target = _normalize_attribute_value(target_value)
                if isinstance(row_value, list):
                    # 리스트인 경우: 리스트 내에 값이 포함되어 있는지 확인
                    return any(_normalize_attribute_value(item) == normalized_target for item in row_value)
                else:
                    # 문자열인 경우: 직접 비교 (대소문자 무시)
                    return _normalize_attribute_value(row_value) == normalized_target
            
            # 필터링 적용
            mask = item_feat[condition].apply(lambda x: matches_attribute(x, attribute_value))
            filtered_items = item_feat[mask]
            
            if len(filtered_items) > 0:
                # 필터링된 아이템의 외부 ID 가져오기
                try:
                    # pandas Series의 .values 속성 사용
                    iid_series = filtered_items[dataset_obj.iid_field]
                    iid_values = iid_series.values
                    # values가 callable인 경우 (함수 객체인 경우) 처리
                    if callable(iid_values):
                        print(f"[경고] iid_series.values가 메서드입니다. to_numpy()를 사용합니다.")
                        filtered_iids = iid_series.to_numpy().tolist()
                    else:
                        filtered_iids = iid_values.tolist()
                except (AttributeError, TypeError) as e:
                    # .values가 없는 경우 또는 오류 발생 시
                    print(f"[경고] iid_series.values 접근 실패: {str(e)}. to_numpy()를 시도합니다.")
                    if hasattr(filtered_items[dataset_obj.iid_field], 'to_numpy'):
                        filtered_iids = filtered_items[dataset_obj.iid_field].to_numpy().tolist()
                    else:
                        filtered_iids = list(filtered_items[dataset_obj.iid_field])
                
                # 외부 ID를 내부 ID로 변환
                filtered_iid_internal = dataset_obj.token2id(dataset_obj.iid_field, filtered_iids)
                
                if len(filtered_iid_internal) > 0:
                    # 필터링된 아이템에 대한 점수만 선택
                    # all_scores의 인덱스는 내부 ID와 일치해야 함
                    # filtered_iid_internal을 텐서로 변환
                    filtered_iid_tensor = torch.tensor(filtered_iid_internal, device=config["device"], dtype=torch.long)
                    filtered_scores = all_scores[:, filtered_iid_tensor]  # shape: [batch_size, num_filtered]
                    
                    # TopK 선택
                    k = min(topK, len(filtered_iid_internal))
                    topk_scores_filtered, topk_indices_filtered = torch.topk(filtered_scores, k=k, dim=1)
                    
                    # 필터링된 인덱스를 원래 아이템 인덱스로 변환
                    topk_iid_list = filtered_iid_tensor[topk_indices_filtered]
                    topk_score = topk_scores_filtered
                    
                    print(f"[필터링] {condition}='{attribute_value}' 조건으로 {len(filtered_iids)}개 아이템 중 {k}개 선택")
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
    external_item_list = dataset_obj.id2token(dataset_obj.iid_field, topk_iid_list.cpu())
    # print(external_item_list)
    external_item_list_name = []
    for u_list in external_item_list:
        external_item_list_name.append([itemID_name.get(iid, '') for iid in u_list])
    external_item_list_name = np.array(external_item_list_name)


    return topk_score, external_item_list, external_item_list_name

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
    for n in range(item_id.shape[0]):
        item_strings = ""
        for s, iid, ina in zip(score[n], item_id[n], item_name[n]):
            item_strings = item_strings + str(iid) + ', ' + str(ina) + ", " + str(round(s.item(), 4)) + "\n"
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
"""
MIND 테스트 데이터셋의 5개 후보를 LLM reranking으로 평가하는 스크립트
- TSV 파일의 세 번째 컬럼에 있는 5개 후보 중 첫 번째가 groundtruth
- LLM에게 reranking 요청하여 재정렬
- MRR, nDCG@5, HIT@1 성능 계산
"""

import json
import pickle
import re
import time
import torch
import numpy as np
from chat_api import llm_chat
from utils import (
    prompt_pattern, user_profile, itemID_name, 
    item_token_id, item_id_token,
    extract_and_check_cur_user_reclist
)
from recbole.evaluator import Evaluator


def parse_tsv_file(tsv_file):
    """
    TSV 파일을 읽어서 사용자별로 5개 후보 추출
    Returns:
        dict: {user_id: {'history': [...], 'candidates': [...], 'groundtruth': '...'}}
    """
    data = {}
    with open(tsv_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                user_id = parts[0]
                history = parts[1].split()  # 두 번째 컬럼: 사용자 히스토리
                candidates = parts[2].split()  # 세 번째 컬럼: 5개 후보
                
                if len(candidates) >= 5:
                    data[user_id] = {
                        'history': history,
                        'candidates': candidates[:5],  # 5개만 사용
                        'groundtruth': candidates[0]  # 첫 번째가 groundtruth
                    }
    return data


def format_candidate_list(candidates, itemID_name):
    """
    5개 후보를 LLM에게 전달할 형식으로 포맷팅
    Format: <ID>, Title, score
    """
    formatted_list = []
    for i, item_id in enumerate(candidates):
        # 외부 ID에서 N 제거 (N104644 -> 104644)
        clean_id = item_id.replace('N', '') if item_id.startswith('N') else item_id
        
        # 아이템 이름 가져오기
        title = itemID_name.get(clean_id, f'News {clean_id}')
        
        # 초기 점수는 순서대로 (5, 4, 3, 2, 1)
        score = 5.0 - i * 0.5
        
        formatted_list.append(f"<{clean_id}>, {title}, {score}")
    
    return '\n'.join(formatted_list)


def format_user_history(user_id, history, itemID_name):
    """
    사용자 히스토리를 포맷팅
    """
    if user_id not in user_profile:
        # user_profile에 없으면 히스토리에서 직접 생성
        history_str = ""
        for item_id in history[-10:]:  # 최근 10개만 사용
            clean_id = item_id.replace('N', '') if item_id.startswith('N') else item_id
            title = itemID_name.get(clean_id, f'News {clean_id}')
            history_str += f"ID <{clean_id}>, {title}.\n"
        return history_str
    else:
        return user_profile[user_id]


def rerank_with_llm(user_id, candidates, history, topK=5):
    """
    LLM을 사용하여 5개 후보를 reranking
    """
    # 후보 리스트 포맷팅
    candidate_list = format_candidate_list(candidates, itemID_name)
    
    # 사용자 히스토리 포맷팅
    user_history = format_user_history(user_id, history, itemID_name)
    
    # Prompt 구성
    instruction = prompt_pattern['knowledge_instruction_2']
    crs_prompt = prompt_pattern['crs_k'].format(
        topK=5, 
        attribute='None', 
        rec_list=candidate_list
    )
    rerank_prompt = prompt_pattern['rerank_default_2'].format(
        before_topK=5, 
        after_topK=topK
    )
    
    question = user_history + crs_prompt + rerank_prompt
    
    # LLM 호출
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            reranked_result = llm_chat(User_message=instruction + question, timeout=60)
            time.sleep(2)  # API 호출 간격
            
            if reranked_result:
                # 마크다운 코드 블록 제거
                if '```' in reranked_result:
                    # 코드 블록 내부 추출
                    match = re.search(r'```.*?\[(.*?)\].*?```', reranked_result, re.DOTALL)
                    if match:
                        reranked_result = '[' + match.group(1) + ']'
                    else:
                        # ``` 제거 후 [ ] 찾기
                        cleaned = reranked_result.replace('```', '').strip()
                        if '[' in cleaned and ']' in cleaned:
                            start = cleaned.find('[')
                            end = cleaned.rfind(']') + 1
                            reranked_result = cleaned[start:end]
                        else:
                            # [ ] 없으면 그냥 ``` 제거
                            reranked_result = cleaned
                
                # [ ]로 감싸지 않은 경우 추가
                if not reranked_result.startswith('['):
                    reranked_result = '[' + reranked_result + ']'
                
                # <ID>, title, score 형식에서 ID 추출
                id_matches = re.findall(r'<(\d+)>', reranked_result)
                if id_matches and len(id_matches) >= topK:
                    # 외부 ID를 내부 ID로 변환 시도
                    internal_ids = []
                    for ext_id in id_matches[:topK]:
                        # item_token_id는 {내부_ID: 외부_ID} 형식
                        # 외부 ID를 찾아서 내부 ID 가져오기
                        found = False
                        for internal_id, external_id in item_token_id.items():
                            if str(external_id) == ext_id:
                                internal_ids.append(str(internal_id))
                                found = True
                                break
                        
                        if not found:
                            # 외부 ID를 그대로 사용 (이미 내부 ID일 수도 있음)
                            internal_ids.append(ext_id)
                    
                    # 내부 ID로 재구성하여 검증
                    reranked_result_internal = '[' + '\n'.join([f"<{item_id}>" for item_id in internal_ids[:topK]]) + ']'
                    validation_result = extract_and_check_cur_user_reclist(reranked_result_internal, topk=topK)
                    
                    if validation_result == 0:  # valid
                        # 외부 ID 반환 (원본 후보 리스트와 비교하기 위해)
                        return id_matches[:topK]
                    else:
                        # 검증 실패해도 외부 ID 반환 (후보 리스트와 비교 가능하도록)
                        return id_matches[:topK]
            
            # 재시도
            if attempt < max_attempts - 1:
                print(f"  [경고] Reranking 실패 (시도 {attempt + 1}/{max_attempts}), 재시도...")
                time.sleep(3)
            else:
                print(f"  [경고] Reranking 최대 재시도 횟수 초과, 원본 순서 사용")
                # 실패 시 원본 순서 반환 (첫 번째가 groundtruth)
                return [c.replace('N', '') if c.startswith('N') else c for c in candidates[:topK]]
                
        except Exception as e:
            print(f"  [오류] Reranking 중 오류 발생 (시도 {attempt + 1}/{max_attempts}): {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(3)
            else:
                # 실패 시 원본 순서 반환
                return [c.replace('N', '') if c.startswith('N') else c for c in candidates[:topK]]
    
    # 모든 시도 실패 시 원본 순서 반환
    return [c.replace('N', '') if c.startswith('N') else c for c in candidates[:topK]]


def calculate_metrics_single_user(groundtruth_id, reranked_list):
    """
    단일 사용자에 대한 MRR, nDCG@5, HIT@1 계산
    """
    # groundtruth ID 정규화 (N 제거)
    groundtruth_clean = groundtruth_id.replace('N', '') if groundtruth_id.startswith('N') else groundtruth_id
    
    # reranked_list에서 groundtruth 위치 찾기
    try:
        rank = reranked_list.index(groundtruth_clean) + 1  # 1-based rank
    except ValueError:
        # groundtruth가 reranked_list에 없는 경우
        rank = len(reranked_list) + 1  # 리스트 길이보다 큰 값
    
    # HIT@1: 첫 번째가 groundtruth인지
    hit_at_1 = 1 if rank == 1 else 0
    
    # MRR: 첫 번째 relevant item의 reciprocal rank
    mrr = 1.0 / rank if rank <= len(reranked_list) else 0.0
    
    # nDCG@5: normalized Discounted Cumulative Gain at 5
    # DCG 계산
    dcg = 0.0
    for i, item_id in enumerate(reranked_list[:5]):
        if item_id == groundtruth_clean:
            # relevant item이 i+1 위치에 있음
            dcg = 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
            break
    
    # IDCG@5: ideal DCG (groundtruth가 첫 번째에 있는 경우)
    idcg = 1.0 / np.log2(2)  # log2(2) = 1
    
    # nDCG@5
    ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0
    
    return {
        'hit@1': hit_at_1,
        'mrr': mrr,
        'ndcg@5': ndcg_at_5,
        'rank': rank
    }


def evaluate_reranking(tsv_file, start_idx=0, end_idx=None):
    """
    TSV 파일의 모든 사용자에 대해 reranking 평가 수행
    """
    print("=" * 80)
    print("MIND Reranking 평가 시작")
    print("=" * 80)
    
    # TSV 파일 읽기
    print(f"\n1. TSV 파일 읽기: {tsv_file}")
    data = parse_tsv_file(tsv_file)
    print(f"   - 총 {len(data)}명의 사용자 로드됨")
    
    # 평가 범위 설정
    user_ids = sorted(data.keys())[start_idx:end_idx]
    total_users = len(user_ids)
    print(f"\n2. 평가 범위: {start_idx} ~ {end_idx if end_idx else len(data)} ({total_users}명)")
    
    # 결과 저장
    all_metrics = {
        'hit@1': [],
        'mrr': [],
        'ndcg@5': [],
        'ranks': []
    }
    
    failed_users = []
    
    # 각 사용자별로 평가
    print(f"\n3. Reranking 평가 시작...")
    print("-" * 80)
    
    for idx, user_id in enumerate(user_ids, 1):
        print(f"\n[{idx}/{total_users}] User {user_id} 처리 중...")
        
        try:
            candidates = data[user_id]['candidates']
            groundtruth = data[user_id]['groundtruth']
            history = data[user_id]['history']
            
            print(f"  Groundtruth: {groundtruth}")
            print(f"  후보 리스트: {candidates}")
            
            # LLM reranking 수행
            print(f"  LLM reranking 요청 중...")
            reranked_list = rerank_with_llm(user_id, candidates, history, topK=5)
            print(f"  Reranked 결과: {reranked_list}")
            
            # 메트릭 계산
            metrics = calculate_metrics_single_user(groundtruth, reranked_list)
            print(f"  HIT@1: {metrics['hit@1']}, MRR: {metrics['mrr']:.4f}, nDCG@5: {metrics['ndcg@5']:.4f}, Rank: {metrics['rank']}")
            
            # 결과 저장
            all_metrics['hit@1'].append(metrics['hit@1'])
            all_metrics['mrr'].append(metrics['mrr'])
            all_metrics['ndcg@5'].append(metrics['ndcg@5'])
            all_metrics['ranks'].append(metrics['rank'])
            
        except Exception as e:
            print(f"  [오류] User {user_id} 처리 실패: {str(e)}")
            failed_users.append(user_id)
            # 실패한 사용자는 0점 처리
            all_metrics['hit@1'].append(0)
            all_metrics['mrr'].append(0.0)
            all_metrics['ndcg@5'].append(0.0)
            all_metrics['ranks'].append(6)  # rank 6 (리스트 밖)
        
        # 진행 상황 출력 (10명마다)
        if idx % 10 == 0:
            avg_hit = np.mean(all_metrics['hit@1'])
            avg_mrr = np.mean(all_metrics['mrr'])
            avg_ndcg = np.mean(all_metrics['ndcg@5'])
            print(f"\n  [진행 상황] {idx}/{total_users} 완료")
            print(f"  현재 평균 - HIT@1: {avg_hit:.4f}, MRR: {avg_mrr:.4f}, nDCG@5: {avg_ndcg:.4f}")
    
    # 최종 결과 출력
    print("\n" + "=" * 80)
    print("최종 평가 결과")
    print("=" * 80)
    
    if len(all_metrics['hit@1']) > 0:
        final_hit = np.mean(all_metrics['hit@1'])
        final_mrr = np.mean(all_metrics['mrr'])
        final_ndcg = np.mean(all_metrics['ndcg@5'])
        avg_rank = np.mean(all_metrics['ranks'])
        
        print(f"\n총 평가 사용자 수: {len(all_metrics['hit@1'])}")
        print(f"실패한 사용자 수: {len(failed_users)}")
        print(f"\n평균 성능:")
        print(f"  HIT@1:  {final_hit:.4f}")
        print(f"  MRR:    {final_mrr:.4f}")
        print(f"  nDCG@5: {final_ndcg:.4f}")
        print(f"  평균 Rank: {avg_rank:.2f}")
        
        if failed_users:
            print(f"\n실패한 사용자: {failed_users[:10]}..." if len(failed_users) > 10 else f"\n실패한 사용자: {failed_users}")
    else:
        print("평가할 사용자가 없습니다.")
    
    return all_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MIND 테스트 데이터셋 reranking 평가')
    parser.add_argument('--tsv_file', type=str, default='MIND_test_(1000)_unique_users.tsv',
                       help='TSV 파일 경로')
    parser.add_argument('--start', type=int, default=0,
                       help='시작 인덱스 (기본값: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='종료 인덱스 (기본값: None, 전체)')
    
    args = parser.parse_args()
    
    # 평가 수행
    metrics = evaluate_reranking(args.tsv_file, args.start, args.end)


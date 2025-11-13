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
import random
from chat_api import llm_chat
from utils import (
    prompt_pattern, user_profile, itemID_name, item_profile,
    item_token_id, item_id_token,
    extract_and_check_cur_user_reclist,
    uid_iid
)
from recbole.evaluator import Evaluator


def parse_tsv_file(tsv_file, shuffle_candidates=True):
    """
    TSV 파일을 읽어서 사용자별로 5개 후보 추출
    Args:
        tsv_file: TSV 파일 경로
        shuffle_candidates: True면 후보 리스트를 랜덤하게 섞음 (기본값: True)
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
                    candidates_list = candidates[:5]  # 5개만 사용
                    groundtruth = candidates_list[0]  # 첫 번째가 groundtruth
                    
                    # 후보 리스트를 랜덤하게 섞기
                    if shuffle_candidates:
                        candidates_list = candidates_list.copy()
                        random.shuffle(candidates_list)
                    
                    data[user_id] = {
                        'history': history,
                        'candidates': candidates_list,
                        'groundtruth': groundtruth  # 원본 groundtruth 유지
                    }
    return data


def format_candidate_list(candidates, itemID_name, item_profile=None):
    """
    5개 후보를 LLM에게 전달할 형식으로 포맷팅
    Format: <ID>, Title, category, subcategory, score
    """
    formatted_list = []
    for i, item_id in enumerate(candidates):
        # 외부 ID에서 N 제거 (N104644 -> 104644)
        clean_id = item_id.replace('N', '') if item_id.startswith('N') else item_id
        
        # item_profile이 있으면 사용 (title, category, subcategory 포함)
        if item_profile and clean_id in item_profile:
            # item_profile 형식: "ID <{iid}>, {title}, category is {category}, subcategory is {subcategory}.\n"
            item_info = item_profile[clean_id].strip()
            # 점수 추가
            score = 5.0 - i * 0.5
            formatted_list.append(f"{item_info.strip('.')}, {score}")
        else:
            # item_profile이 없으면 title만 사용
            title = itemID_name.get(clean_id, f'News {clean_id}')
            score = 5.0 - i * 0.5
            formatted_list.append(f"<{clean_id}>, {title}, {score}")
    
    return '\n'.join(formatted_list)


def format_user_history(user_id, history, itemID_name, item_profile=None):
    """
    사용자 히스토리를 포맷팅
    """
    # user_id를 정수와 문자열 모두 시도
    user_id_int = None
    user_id_str = str(user_id)
    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        pass
    
    # user_profile에서 찾기 (정수와 문자열 모두 시도)
    if user_id_int is not None and user_id_int in user_profile:
        return user_profile[user_id_int]
    elif user_id_str in user_profile:
        return user_profile[user_id_str]
    else:
        # user_profile에 없으면 히스토리에서 직접 생성
        history_str = ""
        for item_id in history[-10:]:  # 최근 10개만 사용
            clean_id = item_id.replace('N', '') if item_id.startswith('N') else item_id
            # item_profile이 있으면 사용 (title, category, subcategory 포함)
            if item_profile and clean_id in item_profile:
                history_str += item_profile[clean_id]
            else:
                # item_profile이 없으면 title만 사용
                title = itemID_name.get(clean_id, f'News {clean_id}')
                history_str += f"ID <{clean_id}>, {title}.\n"
        return history_str


def rerank_with_model(user_id, candidates, topK=5, condition='None'):
    """
    학습된 retrieval 모델의 점수를 사용하여 5개 후보를 reranking
    """
    from call_crs import retrieval_topk
    
    try:
        # 5개 후보에 대해 모델 점수 계산
        # 먼저 모든 아이템에 대한 점수를 가져오기 위해 큰 topK로 retrieval
        topk_score, external_item_list, external_item_list_name = retrieval_topk(
            dataset='mind',
            condition=condition,
            user_id=[user_id],
            topK=1000,  # 충분히 큰 값
            mode='freeze'
        )
        
        # 후보 아이템들의 점수 찾기
        candidate_scores = {}
        # external_item_list는 [batch_size, topK] 형태
        if len(external_item_list) > 0:
            item_list = external_item_list[0]  # 첫 번째 사용자
            score_list = topk_score[0].cpu().numpy() if isinstance(topk_score, torch.Tensor) else topk_score[0]
            
            # 후보 아이템 ID 정규화 (N 제거)
            normalized_candidates = [c.replace('N', '') if c.startswith('N') else c for c in candidates]
            
            # 각 후보 아이템의 점수 찾기
            for i, item_id in enumerate(item_list):
                if item_id in normalized_candidates:
                    score = float(score_list[i]) if i < len(score_list) else 0.0
                    candidate_scores[item_id] = score
            
            # 점수가 없는 후보는 0점 처리
            for cand in normalized_candidates:
                if cand not in candidate_scores:
                    candidate_scores[cand] = 0.0
            
            # 점수로 정렬 (내림차순)
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            reranked_list = [item_id for item_id, score in sorted_candidates[:topK]]
            
            return reranked_list
        else:
            # 점수를 찾을 수 없으면 원본 순서 반환
            return [c.replace('N', '') if c.startswith('N') else c for c in candidates[:topK]]
            
    except Exception as e:
        print(f"  [경고] 모델 reranking 실패: {str(e)}, 원본 순서 사용")
        return [c.replace('N', '') if c.startswith('N') else c for c in candidates[:topK]]


def rerank_with_llm(user_id, candidates, history, topK=5):
    """
    LLM을 사용하여 5개 후보를 reranking (LLM의 일반 지식 사용)
    """
    # 후보 리스트 포맷팅
    candidate_list = format_candidate_list(candidates, itemID_name, item_profile)
    
    # 사용자 히스토리 포맷팅
    user_history = format_user_history(user_id, history, itemID_name, item_profile)
    
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


def evaluate_reranking_with_react(tsv_file, start_idx=0, end_idx=None):
    """
    TSV 파일의 모든 사용자에 대해 ReAct 패턴으로 reranking 평가 수행
    - LLM이 5개 후보를 보고 만족스러운지 판단
    - 만족스럽지 않으면 reranking tool 사용
    - 만족스럽으면 finish
    """
    from chat_recEnv import RecEnv
    from chat_recWrappers import reActWrapper, LoggingWrapper
    import chat_recEnv, chat_recWrappers
    
    # 환경 설정
    env = chat_recEnv.RecEnv()
    env = chat_recWrappers.reActWrapper(env)
    env = chat_recWrappers.LoggingWrapper(env)
    
    print("=" * 80)
    print("MIND Reranking 평가 시작 (ReAct 패턴)")
    print("=" * 80)
    
    # TSV 파일 읽기
    print(f"\n1. TSV 파일 읽기: {tsv_file}")
    data = parse_tsv_file(tsv_file, shuffle_candidates=True)
    print(f"   - 총 {len(data)}명의 사용자 로드됨")
    print(f"   - 후보 리스트는 랜덤하게 섞어서 제공됩니다")
    
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
    print(f"\n3. ReAct 패턴으로 Reranking 평가 시작...")
    print("-" * 80)
    
    for idx, user_id in enumerate(user_ids, 1):
        print(f"\n[{idx}/{total_users}] User {user_id} 처리 중...")
        
        try:
            candidates = data[user_id]['candidates']
            groundtruth = data[user_id]['groundtruth']
            history = data[user_id]['history']
            
            print(f"  Groundtruth: {groundtruth}")
            print(f"  후보 리스트: {candidates}")
            
            # 5개 후보를 초기 observation으로 설정
            candidate_list = format_candidate_list(candidates, itemID_name, item_profile)
            
            # 환경 초기화 (user_profile에 있는 키 형식으로 변환)
            # user_profile의 키 형식 확인 및 변환
            user_id_for_env = None
            user_id_str = str(user_id)
            try:
                user_id_int = int(user_id)
            except (ValueError, TypeError):
                user_id_int = None
            
            # user_profile에 있는 키 형식 찾기
            if user_id_int is not None and user_id_int in user_profile:
                user_id_for_env = user_id_int
            elif user_id_str in user_profile:
                user_id_for_env = user_id_str
            else:
                # user_profile에 없으면 uid_iid에서 찾기
                if user_id_int is not None and user_id_int in uid_iid:
                    user_id_for_env = user_id_int
                elif user_id_str in uid_iid:
                    user_id_for_env = user_id_str
                else:
                    # 여전히 없으면 첫 번째 사용 가능한 user_id 사용
                    print(f"  [경고] User {user_id}가 user_profile에 없습니다. 첫 번째 사용자 사용.")
                    if len(uid_iid) > 0:
                        user_id_for_env = list(uid_iid.keys())[0]
                    else:
                        raise ValueError(f"User {user_id}를 찾을 수 없고, 사용 가능한 사용자도 없습니다.")
            
            # user_profile에 없으면 임시로 추가 (히스토리에서 생성)
            if user_id_for_env not in user_profile:
                print(f"  [경고] User {user_id_for_env}가 user_profile에 없습니다. 히스토리에서 생성합니다.")
                user_history_str = format_user_history(user_id, history, itemID_name, item_profile)
                user_profile[user_id_for_env] = user_history_str
            
            env.reset(userID=user_id_for_env)
            
            # 5개 후보를 rec_traj에 추가 (CRS 결과로 가정)
            # rerank_step에서 이전 결과를 참조하므로 rec_traj에 추가
            env.rec_traj.append(['crs', '5', 'None', candidate_list])
            
            # 초기 observation 설정 (5개만 있다는 것을 명확히 표시)
            # LLM이 attribute를 자유롭게 선택할 수 있도록 안내
            initial_obs = f"Here are **exactly 5 candidate news articles** from the recommender system (no more, no less):\n{candidate_list}\n\n**IMPORTANT: You have exactly 5 candidate articles. Please use Rerank[attribute, 5] to rerank these 5 articles, NOT 10.**\n\nYou can use different attributes for reranking:\n- Rerank[None, 5]: Rerank without any attribute condition\n- Rerank[category, 5]: Rerank based on category attribute\n- Rerank[subcategory, 5]: Rerank based on subcategory attribute\n\nPlease evaluate if this ranking is satisfactory. If not, use Rerank tool with an appropriate attribute to improve it. Remember: you can only rerank the 5 articles provided above."
            env.obs = initial_obs
            
            # ReAct 패턴으로 LLM이 판단하고 reranking tool 사용
            # 최대 3번의 action 시도 (retrieve/rerank/finish)
            max_steps = 3
            final_list = None
            
            for step in range(max_steps):
                # LLM에게 Thought와 Action 요청
                user_history = format_user_history(user_id, history, itemID_name, item_profile)
                current_obs = env.obs
                
                # Prompt 구성
                instruction = prompt_pattern['instruction']
                task_prompt = prompt_pattern['task']
                question = user_history + current_obs + task_prompt
                
                # LLM 호출
                try:
                    thought_action = llm_chat(
                        User_message=instruction + question + f"Thought {step+1}:",
                        timeout=60
                    )
                    time.sleep(2)
                    
                    # Thought와 Action 파싱
                    if f"\nAction {step+1}:" in thought_action:
                        thought, action = thought_action.strip().split(f"\nAction {step+1}:")
                    else:
                        thought = thought_action.strip().split('\n')[0]
                        action = llm_chat(
                            User_message=instruction + question + f"Thought {step+1}: {thought}\nAction {step+1}:",
                            timeout=60
                        ).strip()
                    
                    # Action에서 실제 액션 부분만 추출
                    action_clean = action.strip()
                    action_match = re.search(r'(retrieve|rerank|finish)\[([^\]]*)\]', action_clean, re.IGNORECASE)
                    if action_match:
                        action_type = action_match.group(1).lower()
                        action_params = action_match.group(2)
                        action_clean = f"{action_type}[{action_params}]"
                    
                    print(f"  Step {step+1}: {thought[:100]}...")
                    print(f"  Action: {action_clean}")
                    
                    # Environment step 실행
                    obs, reward, done, info = env.step(action_clean)
                    
                    if done:
                        # Finish action이 실행됨
                        final_list = info.get('answer', '')
                        print(f"  Finish! Final list: {final_list[:200]}...")
                        break
                    else:
                        print(f"  Observation: {obs[:200]}...")
                        
                except Exception as e:
                    print(f"  [오류] Step {step+1} 실패: {str(e)}")
                    import traceback
                    print(f"  [상세 오류] {traceback.format_exc()}")
                    break
            
            # 최종 리스트에서 아이템 ID 추출
            if final_list:
                id_matches = re.findall(r'<(\d+)>', final_list)
                reranked_list = id_matches[:5] if len(id_matches) >= 5 else id_matches
            else:
                # 실패 시 원본 순서 사용
                reranked_list = [c.replace('N', '') if c.startswith('N') else c for c in candidates[:5]]
            
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
            import traceback
            print(f"  [오류] User {user_id} 처리 실패: {str(e)}")
            print(f"  [상세 오류] {traceback.format_exc()}")
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


def evaluate_reranking(tsv_file, start_idx=0, end_idx=None, use_model=False, condition='None'):
    """
    TSV 파일의 모든 사용자에 대해 reranking 평가 수행
    
    Args:
        tsv_file: TSV 파일 경로
        start_idx: 시작 인덱스
        end_idx: 종료 인덱스
        use_model: True면 학습된 모델 사용, False면 LLM 사용
        condition: 모델 사용 시 attribute 조건 ('None', 'category', 'subcategory')
    """
    print("=" * 80)
    print("MIND Reranking 평가 시작")
    print("=" * 80)
    
    # TSV 파일 읽기
    print(f"\n1. TSV 파일 읽기: {tsv_file}")
    data = parse_tsv_file(tsv_file, shuffle_candidates=True)
    print(f"   - 총 {len(data)}명의 사용자 로드됨")
    print(f"   - 후보 리스트는 랜덤하게 섞어서 제공됩니다")
    
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
            
            # Reranking 수행
            if use_model:
                print(f"  학습된 모델 reranking 수행 중... (condition: {condition})")
                reranked_list = rerank_with_model(user_id, candidates, topK=5, condition=condition)
            else:
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
    parser.add_argument('--use_model', action='store_true',
                       help='학습된 모델 사용 (기본값: False, LLM 사용)')
    parser.add_argument('--condition', type=str, default='None',
                       choices=['None', 'category', 'subcategory'],
                       help='모델 사용 시 attribute 조건 (기본값: None)')
    parser.add_argument('--use_react', action='store_true',
                       help='ReAct 패턴 사용 (LLM이 자동으로 판단하고 reranking tool 사용)')
    
    args = parser.parse_args()
    
    # 평가 수행
    if args.use_react:
        # ReAct 패턴 사용 (원래 프로젝트 방식)
        metrics = evaluate_reranking_with_react(args.tsv_file, args.start, args.end)
    else:
        # 직접 reranking (기존 방식)
        metrics = evaluate_reranking(args.tsv_file, args.start, args.end, 
                                     use_model=args.use_model, condition=args.condition)


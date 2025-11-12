from chat_api import llm_chat
import time


import chat_recEnv, chat_recWrappers
# instruct to call recsys
env = chat_recEnv.RecEnv()
env = chat_recWrappers.reActWrapper(env)
env = chat_recWrappers.LoggingWrapper(env)

def step(env, action):
    attemps = 0
    max_attempts = 10
    while attemps < max_attempts:
        try:
            return env.step(action)
        except Exception as e:
            # Print the details of the exception
            print(f"Step 실행 오류 (시도 {attemps + 1}/{max_attempts}): {str(e)}")
            attemps += 1
            if attemps < max_attempts:
                time.sleep(2)  # 재시도 전 대기
            else:
                print("Step 실행 최대 재시도 횟수 초과")
                raise


import json
import sys
import os
import pickle
import argparse
import gc
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="Cycle start.")
parser.add_argument("--step_num", type=int, default=100, help="Cycle step number.")
args, _ = parser.parse_known_args()



task_prompt = prompt_dict['think_sample']
instruction = prompt_pattern['instruction']

question_tail = prompt_pattern['task']

def task_customization(userID, sys_role=instruction, prompt=task_prompt, to_print=True):
    question = env.reset(userID=userID)     # return 
    if to_print:
        print(userID, question)
    question = prompt + question + question_tail
    n_calls, n_badcalls, bad_flag = 0, 0, 0

    for i in range(1, 8):
        n_calls += 1
        if to_print:
            print(f"[User {userID}] Step {i}/7 진행 중...")

        try:
            # question이 너무 길어지면 메모리 절약을 위해 일부만 유지
            current_question = sys_role + question + f"Thought {i}:"
            if len(current_question) > 10000:  # 10KB 이상이면 최근 부분만 유지
                # 최근 5000자만 유지
                question = question[-5000:] + question_tail
            
            thought_action = llm_chat(User_message=current_question, stop=f"\nObservation {i}:", timeout=60)
            time.sleep(2)
        except Exception as e:
            print(f"[User {userID}] Step {i} API 호출 실패: {str(e)}")
            raise

        try:
            # try again
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            bad_flag = 1
            thought = thought_action.strip().split('\n')[0]
            action = llm_chat(User_message=sys_role+question + f"Thought {i}: {thought}\nAction {i}:", stop=f"\n").strip()

        if f"\nObservation {i}: " in action:
            action = action.strip().split(f"\nObservation {i}: ")[0]
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        question += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt, 'user_idx': userID})
    return r, info



num_user = len(uid_iid)

import time
import random
import logging

# 로깅 설정: reentrant call 방지를 위해 안전하게 설정
os.makedirs('./trajs', exist_ok=True)

# OpenAI 및 httpcore의 로깅 레벨을 WARNING으로 설정하여 디버그 로그 비활성화
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 메인 로거 설정
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 기존 핸들러 제거
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 파일 핸들러 추가 (thread-safe)
try:
    file_handler = logging.FileHandler('./trajs/828.log', mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    # 로깅 설정 실패 시 무시하고 계속 진행
    print(f"Warning: Could not set up logging: {e}")


reward_s = []

old_time = time.time

failed_times = 0
flag_run = 1
u_num = 0



infos = []
save_interval = 10  # 10명마다 중간 저장 및 메모리 정리

for uid in uid_iid.keys():

    u_num += 1

    if u_num < args.start:
        continue
    if u_num > args.start + args.step_num:
        break


    try: 
        start_time = time.time()
        print(f"\n[진행 상황] User {u_num}/{args.start + args.step_num} 처리 중 (User ID: {uid})")
        r, info = task_customization(uid, to_print=True)
        elapsed_time = time.time() - start_time
        infos.append(info)
        print('steps, \t recsys_steps, \t llm_steps, \t answer')
        logging.info('steps {step}, \t recsys_steps {recsys_steps}, \t llm_steps {llm_steps}, \n answer {answer} \n trajectory {traj}'.format(step=info['steps'], recsys_steps=info['recsys_steps'], llm_steps=info['llm_steps'], answer=info['answer'], traj=info['rec_traj']))
        print(f'[완료] User {uid} 처리 완료 (소요 시간: {elapsed_time:.2f}초)')
        print('-----------')
        
        # 주기적으로 중간 저장 및 메모리 정리
        if u_num % save_interval == 0:
            print(f"[메모리 최적화] {u_num}명 처리 완료 - 중간 저장 및 메모리 정리 중...")
            # 중간 저장
            output_dir = f"chat_his/{dataset_name}"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"chat_his/{dataset_name}/start_{args.start}_intermediate_{u_num}.json", 'w') as f:
                json.dump(infos, f)
            # 메모리 정리
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[메모리 최적화] 중간 저장 완료 (현재까지 {len(infos)}개 결과 저장)")
            
    except Exception as e:
        failed_times += 1
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        # Print the details of the exception
        print(f"[오류] User {uid} 처리 실패 (소요 시간: {elapsed_time:.2f}초)")
        print("An error occurred:", str(e))
        print("OHHHHHHHH... User {user} Failed. Failed_times is {fail}".format(user=uid, fail=failed_times))
        time.sleep(20)
        # 오류 발생 시에도 메모리 정리
        gc.collect()


# 결과 저장 디렉토리 생성
output_dir = f"chat_his/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)

with open("chat_his/{dataset}/start_{st}.json".format(dataset=dataset_name, st=args.start), 'w') as f:   
    json.dump(infos, f)
    print("====  Info Dump Ends ====")

# 성능 평가 수행
print("\n" + "="*50)
print("성능 평가 시작...")
print("="*50)

try:
    from utils import evaluate_user, item_token_id, user_token_id, user_id_token, uid_iid
    
    # MIND 데이터셋의 item_num 설정
    if dataset_name == 'mind':
        item_num = len(item_token_id)
    elif dataset_name == 'yelp':
        item_num = 45582
    elif dataset_name == 'ml-1m':
        item_num = 3884
    elif dataset_name == 'amazon_book':
        item_num = 97566
    else:
        item_num = len(item_token_id)
    
    # 추천 리스트 추출
    uid_topK = {}
    valid_count = 0
    invalid_count = 0
    
    for info in infos:
        # user_idx 또는 user_id 필드 확인
        user_idx = str(info.get('user_idx', '')) or str(info.get('user_id', ''))
        if not user_idx:
            # info에 user_id가 없으면 스킵 (이론적으로는 있어야 함)
            print(f"경고: user_idx/user_id가 없는 info 발견: {list(info.keys())}")
            continue
            
        # answer에서 추천 리스트 추출
        answer = info.get('answer', '').strip()
        if answer:
            res_tmp = [item.split(', ')[0] for item in answer.split('\n') if item.strip()]
            if len(res_tmp) >= 10:
                uid_topK[user_idx] = res_tmp[:10]
                valid_count += 1
            else:
                # rec_traj에서 추천 리스트 추출 시도
                rec_traj = info.get('rec_traj', [])
                found = False
                for tj in reversed(rec_traj):
                    if isinstance(tj, list) and len(tj) >= 4:
                        if tj[0] == 'rerank' and str(tj[1]).isnumeric() and int(tj[1]) >= 10:
                            try:
                                traj_items = tj[3].strip()
                                if traj_items.startswith('[') and traj_items.endswith(']'):
                                    traj_items = traj_items[1:-1].strip()
                                item_list = [item.split(",")[0].strip() for item in traj_items.split("\n") if item.strip()]
                                if len(item_list) >= 10:
                                    uid_topK[user_idx] = item_list[:10]
                                    valid_count += 1
                                    found = True
                                    break
                            except:
                                pass
                if not found:
                    invalid_count += 1
        else:
            invalid_count += 1
    
    print(f"전체 사용자: {len(infos)}, 유효한 추천: {valid_count}, 무효한 추천: {invalid_count}")
    
    if len(uid_topK) == 0:
        print("경고: 평가할 유효한 추천 리스트가 없습니다.")
    else:
        # 아이템 ID 검증 및 필터링
        count_out_index = 0
        uid_topK_filtered = {}
        for uid in list(uid_topK.keys()):
            valid_items = []
            for iid in uid_topK[uid]:
                if item_token_id.get(iid, 0):
                    valid_items.append(iid)
            if len(valid_items) >= 10:
                uid_topK_filtered[uid] = valid_items[:10]
            else:
                count_out_index += 1
        
        print(f"아이템 ID 검증 후: 유효 {len(uid_topK_filtered)}명, 제외 {count_out_index}명")
        
        if len(uid_topK_filtered) > 0:
            # 사용자 및 아이템 ID 매핑
            pos_user_before_map = [user_token_id[uid] for uid in uid_topK_filtered.keys() if uid in user_token_id]
            pos_user_before_map.sort()
            pos_user_list_str = [user_id_token[uid] for uid in pos_user_before_map]
            
            user_num = len(pos_user_list_str)
            pos_user_list = list(range(user_num))
            pos_item_list = [item_token_id[uid_iid[uid]] for uid in pos_user_list_str if uid in uid_iid and uid_iid[uid] in item_token_id]
            topk_idx_list = [[item_token_id[iid] for iid in uid_topK_filtered[uid] if iid in item_token_id] for uid in pos_user_list_str if uid in uid_topK_filtered]
            
            # 리스트 길이 맞추기
            min_len = min(len(pos_user_list), len(pos_item_list), len(topk_idx_list))
            pos_user_list = pos_user_list[:min_len]
            pos_item_list = pos_item_list[:min_len]
            topk_idx_list = topk_idx_list[:min_len]
            
            # topk_idx_list의 각 항목이 10개인지 확인
            topk_idx_list_filtered = []
            pos_user_list_filtered = []
            pos_item_list_filtered = []
            for i, topk in enumerate(topk_idx_list):
                if len(topk) == 10:
                    topk_idx_list_filtered.append(topk)
                    pos_user_list_filtered.append(pos_user_list[i])
                    pos_item_list_filtered.append(pos_item_list[i])
            
            if len(topk_idx_list_filtered) > 0:
                # 평가 수행
                chat_eval_result = evaluate_user(
                    pos_user_list_filtered, 
                    pos_item_list_filtered, 
                    topk_idx_list_filtered, 
                    len(topk_idx_list_filtered), 
                    item_num
                )
                
                print("\n" + "="*50)
                print("최종 평가 결과:")
                print("="*50)
                for metric, value in chat_eval_result.items():
                    print(f"{metric}: {value:.4f}")
                print("="*50)
            else:
                print("경고: 유효한 추천 리스트가 없습니다 (각 리스트가 10개 아이템을 포함해야 함).")
        else:
            print("경고: 평가할 유효한 추천 리스트가 없습니다.")
            
except Exception as e:
    print(f"평가 중 오류 발생: {str(e)}")
    import traceback
    traceback.print_exc()
    print("\n수동으로 평가하려면 다음 명령을 실행하세요:")
    print(f"python chat_analysis.py")

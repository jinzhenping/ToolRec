from chat_api import llm_chat
import time


import chat_recEnv, chat_recWrappers
# instruct to call recsys
env = chat_recEnv.RecEnv()
env = chat_recWrappers.reActWrapper(env)
env = chat_recWrappers.LoggingWrapper(env)

def step(env, action):
    attemps = 0
    while attemps < 10:
        try:
            return env.step(action)
        except Exception as e:
            # Print the details of the exception
            print("An error occurred:", str(e))
            attemps += 1


import json
import sys
import os
import pickle
import argparse
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

        thought_action = llm_chat(User_message=sys_role+question+f"Thought {i}:", stop=f"\nObservation {i}:")
        time.sleep(2)

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
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
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
for uid in uid_iid.keys():

    u_num += 1

    if u_num < args.start:
        continue
    if u_num > args.start + args.step_num:
        break


    try: 
        r, info = task_customization(uid, to_print=True)
        infos.append(info)
        print('steps, \t recsys_steps, \t llm_steps, \t answer')
        logging.info('steps {step}, \t recsys_steps {recsys_steps}, \t llm_steps {llm_steps}, \n answer {answer} \n trajectory {traj}'.format(step=info['steps'], recsys_steps=info['recsys_steps'], llm_steps=info['llm_steps'], answer=info['answer'], traj=info['rec_traj']))
        print('-----------')
    except Exception as e:
        failed_times += 1
        # Print the details of the exception
        time.sleep(20)
        print("An error occurred:", str(e))
        print("OHHHHHHHH... User {user} Failed. Failed_times is {fail}".format(user=uid, fail=failed_times))


# 결과 저장 디렉토리 생성
output_dir = f"chat_his/{dataset_name}"
os.makedirs(output_dir, exist_ok=True)

with open("chat_his/{dataset}/start_{st}.json".format(dataset=dataset_name, st=args.start), 'w') as f:   
    json.dump(infos, f)
    print("====  Info Dump Ends ====")

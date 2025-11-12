import ast
import pickle
import json
import time
import gym
import requests
from bs4 import BeautifulSoup
from call_crs import retrieval_topk, stdout_retrived_items
from chat_api import llm_chat

from utils import *
import time


class textSpace(gym.spaces.Space):
  def contains(self, x) -> bool:
    """Return boolean specifying if x is a valid member of this space."""
    return isinstance(x, str)


class RecEnv(gym.Env):

    def __init__(self):
        """
        Initialize the environment.
        """
        super().__init__()
        self.obs = None  # current observation
        self.steps = 0  # current number of steps
        self.answer = None  # current answer from the agent
        self.observation_space = self.action_space = textSpace()
        self.num_retrieval = 0
        self.num_rerank = 0
        # self.send_chat = ssh_chat()
    
    def _get_obs(self):
        return self.obs

    def _get_info(self):
        return {"steps": self.steps, "recsys_steps":self.call_Recsys_cnt, "llm_steps":self.call_llm_cnt, "answer": self.answer, "rec_traj":[step for step in self.rec_traj]}

    def reset(self, seed=None, return_info=False, options=None, uid=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)
        self.obs = ("Interact with RecSys/LLM using retrieve[], rerank[], and finish[].\n")
        self.user_id = uid
        self.rec_traj = []
        self.page = None
        self.condition_keyword = None
        self.condition_list = None
        self.call_Recsys_cnt = 0
        self.call_llm_cnt = 0
        self.steps = 0
        self.answer = None
        self.final_length = 10
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
    
    def retrieval_step(self, attribute, topK):
        if attribute in DATASET_ATT:
            uid = [self.user_id]
            topk_score, external_item_list, external_item_list_name = retrieval_topk(dataset=dataset_name, condition=attribute, user_id=uid, topK=topK)
            retrived_items = stdout_retrived_items(topk_score, external_item_list, external_item_list_name)
            item_list = '[' + retrived_items[0] + ']'
            self.call_Recsys_cnt += 1
            self.rec_traj.append(['crs', topK, attribute, item_list])
            self.obs = item_list
        else:
            # retrieval from LLM, and or rerank....
            self.rerank_step(attribute, topK)

    
    def rerank_step(self, attribute, topK):
        instruction = prompt_pattern['knowledge_instruction_2']
        # examples = prompt_dict['ranking_sample']
        self.user_profile = user_profile[self.user_id]
        prompt_cur = ''
        for line in self.rec_traj:
            if line[0] == 'crs':
                prompt_cur = prompt_cur + prompt_pattern['crs_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
            elif line[0] == 'rerank':
                prompt_cur = prompt_cur + prompt_pattern['rerank_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
        last_mode = self.rec_traj[-1][0]
        if attribute == "None":
            previous_topK = sum([int(i[1]) for i in self.rec_traj])
            prompt_output = prompt_pattern['rerank_default_2'].format(before_topK=previous_topK, after_topK=topK)
        else:
            previous_topK = sum([int(i[1]) for i in self.rec_traj])
            prompt_output = prompt_pattern['rerank_output_2'].format(before_topK=previous_topK, rerank_type=attribute, after_topK=topK)
        
        question = user_profile[self.user_id] + prompt_cur + prompt_output
        attemps = 0
        max_attempts = 10
        reranked_result = None
        
        while attemps < max_attempts:
            attemps += 1
            try:
                # reranked_result = llm_chat(role=instruction, User_message=question)
                reranked_result = llm_chat(User_message=instruction+question, timeout=60)
                time.sleep(4)
                
                # LLM 응답에서 실제 리스트 부분만 추출 (마크다운 코드 블록 제거)
                if reranked_result:
                    import re
                    # ``` 로 감싸진 부분 제거
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
                    
                    # <ID>, title, score 형식인 경우 파싱
                    if '<' in reranked_result and '>' in reranked_result:
                        # <42208>, Bold predictions..., 7.5 형식 파싱
                        id_matches = re.findall(r'<(\d+)>', reranked_result)
                        if id_matches and len(id_matches) >= topK:
                            # 외부 ID를 내부 ID로 변환 시도 (필요한 경우)
                            # 먼저 외부 ID 그대로 사용 시도
                            # 만약 실패하면 내부 ID로 변환
                            reranked_result = '[' + '\n'.join([f"<{item_id}>" for item_id in id_matches[:topK]]) + ']'
                    
                    if not reranked_result.startswith("["):
                        reranked_result = '[' + reranked_result + ']'

                # LLM 응답 검증 및 외부 ID -> 내부 ID 변환 시도
                if reranked_result and not extract_and_check_cur_user_reclist(reranked_result, topk=topK):
                    # 외부 ID를 내부 ID로 변환 시도
                    import re
                    id_matches = re.findall(r'<(\d+)>', reranked_result)
                    if id_matches and len(id_matches) >= topK:
                        # item_id_token을 사용하여 외부 ID -> 내부 ID 변환
                        try:
                            from utils import item_id_token
                            internal_ids = []
                            for ext_id in id_matches[:topK]:
                                # item_id_token은 외부 ID -> 내부 ID 매핑
                                if ext_id in item_id_token:
                                    internal_ids.append(str(item_id_token[ext_id]))
                                else:
                                    # item_id_token에 없으면 외부 ID 그대로 사용
                                    internal_ids.append(ext_id)
                            
                            if len(internal_ids) >= topK:
                                # 내부 ID로 재구성
                                reranked_result_internal = '[' + '\n'.join([f"<{item_id}>" for item_id in internal_ids[:topK]]) + ']'
                                if extract_and_check_cur_user_reclist(reranked_result_internal, topk=topK):
                                    print(f"  [정보] 외부 ID를 내부 ID로 변환 성공")
                                    reranked_result = reranked_result_internal
                                    break
                        except Exception as e:
                            print(f"  [경고] ID 변환 시도 중 오류: {str(e)}")
                    
                    # 이전 rec_traj에서 아이템 리스트 가져오기
                    previous_items = None
                    for line in reversed(self.rec_traj):
                        if line[0] == 'crs' and len(line) >= 4:
                            previous_items = line[3]
                            break
                    
                    if previous_items:
                        # LLM 응답에서 제목 추출
                        import re
                        # 제목 패턴 추출 (예: "'One" 'in' 'a' "million'" ...)
                        title_patterns = re.findall(r"['\"]([^'\"]+)['\"]", reranked_result)
                        
                        # 이전 결과에서 아이템 ID와 제목 매핑
                        previous_lines = str(previous_items).strip('[]').split('\n')
                        item_id_to_title = {}
                        for prev_line in previous_lines:
                            prev_line = prev_line.strip()
                            if not prev_line:
                                continue
                            # <ID>, [title...], score 형식 파싱
                            id_match = re.search(r'<(\d+)>', prev_line)
                            if id_match:
                                item_id = id_match.group(1)
                                # 제목 부분 추출 (대괄호 안)
                                title_match = re.search(r'\[(.*?)\]', prev_line)
                                if title_match:
                                    title_text = title_match.group(1)
                                    # 제목 정규화 (공백, 따옴표 제거)
                                    title_normalized = re.sub(r"['\"]", '', title_text).strip()
                                    item_id_to_title[item_id] = title_normalized
                        
                        # LLM 응답의 제목과 매칭되는 아이템 ID 찾기
                        matched_ids = []
                        for title_pattern in title_patterns[:topK]:
                            title_normalized = re.sub(r"['\"]", '', title_pattern).strip()
                            # 부분 매칭으로 찾기
                            for item_id, stored_title in item_id_to_title.items():
                                if title_normalized.lower() in stored_title.lower() or stored_title.lower() in title_normalized.lower():
                                    if item_id not in matched_ids:
                                        matched_ids.append(item_id)
                                        break
                        
                        # 매칭된 아이템 ID로 리스트 재구성
                        if len(matched_ids) >= topK:
                            # 이전 결과에서 순서대로 가져오기
                            final_ids = []
                            for prev_line in previous_lines:
                                prev_line = prev_line.strip()
                                if not prev_line:
                                    continue
                                id_match = re.search(r'<(\d+)>', prev_line)
                                if id_match and id_match.group(1) in matched_ids:
                                    final_ids.append(id_match.group(1))
                                    if len(final_ids) >= topK:
                                        break
                            
                            if len(final_ids) >= topK:
                                # 아이템 ID 리스트를 올바른 형식으로 변환
                                reranked_result = '[' + '\n'.join([f"<{item_id}>" for item_id in final_ids[:topK]]) + ']'
                                if extract_and_check_cur_user_reclist(reranked_result, topk=topK):
                                    print(f"  [정보] LLM 응답에서 제목을 아이템 ID로 매핑 성공")
                                    break
                
                if reranked_result and extract_and_check_cur_user_reclist(reranked_result, topk=topK):
                    break
                else:
                    if attemps < max_attempts:
                        print(f"  [경고] Rerank 응답 형식 오류 (시도 {attemps}/{max_attempts}), 재시도 중...")
                        if reranked_result:
                            print(f"  응답 샘플: {reranked_result[:200]}...")
            except Exception as e:
                print(f"Rerank API 호출 오류 (시도 {attemps}/{max_attempts}): {str(e)}")
                if attemps < max_attempts:
                    time.sleep(5)
                else:
                    print("Rerank 최대 재시도 횟수 초과")
                    raise
        
        # 최종 결과가 없으면 이전 결과 재사용
        if not reranked_result or not extract_and_check_cur_user_reclist(reranked_result, topk=topK):
            print(f"  [경고] Rerank 실패: 이전 추천 리스트 재사용")
            # 이전 rec_traj에서 마지막 유효한 리스트 찾기
            for line in reversed(self.rec_traj):
                if line[0] == 'crs' and len(line) >= 4:
                    reranked_result = line[3]
                    break
        self.rec_traj.append(['rerank', topK, attribute, reranked_result])
        self.call_llm_cnt += 1
        self.obs = reranked_result
        return reranked_result

    def conclude_step(self, topK):
        instruction = prompt_pattern['knowledge_instruction_2']
        # examples = prompt_dict['ranking_sample']
        self.user_profile = user_profile[self.user_id]
        prompt_cur = ''
        for line in self.rec_traj:
            if line[0] == 'crs':
                prompt_cur = prompt_cur + prompt_pattern['crs_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
            elif line[0] == 'rerank':
                prompt_cur = prompt_cur + prompt_pattern['rerank_k'].format(topK=line[1], attribute=line[2], rec_list=line[3])
        previous_topK = sum([int(i[1]) for i in self.rec_traj])
        prompt_output = prompt_pattern['rerank_default_2'].format(before_topK=previous_topK, after_topK=topK)

        question = user_profile[self.user_id] + prompt_cur + prompt_output
        attemps = 0
        max_attempts = 10
        while attemps < max_attempts:
            attemps += 1
            try:
                # reranked_result = llm_chat(role=instruction, User_message=question)
                reranked_result = llm_chat(User_message=instruction+question, timeout=60)
                time.sleep(4)
                if not reranked_result.startswith("["):
                    reranked_result = '[' + reranked_result + ']'
                if extract_and_check_cur_user_reclist(reranked_result, topk=topK):
                    break
            except Exception as e:
                print(f"Rerank API 호출 오류 (시도 {attemps}/{max_attempts}): {str(e)}")
                if attemps < max_attempts:
                    time.sleep(5)
                else:
                    print("Rerank 최대 재시도 횟수 초과")
                    raise
        
        self.rec_traj.append(['rerank', topK, ' ', reranked_result])
        self.call_llm_cnt += 1
        self.obs = reranked_result
        return reranked_result
            

    
    def step(self, action):
        '''retrieve[], rerank[], and finish[]'''
        reward = 0
        done = False
        action = action.strip()
        if self.answer is not None:  # already finished
            done = True
            return self.obs, reward, done, self._get_info()
        
        # 액션에서 실제 액션 부분만 추출 (정규식 사용)
        import re
        action_match = re.search(r'(retrieve|rerank|finish)\[([^\]]+)\]', action, re.IGNORECASE)
        if action_match:
            action_type = action_match.group(1).lower()
            action_params = action_match.group(2)
            
            if action_type == "retrieve":
                rec_condition, topN = action_params.split(',')[0].strip(), action_params.split(',')[1].strip()
                topN = int(topN)
                self.retrieval_step(rec_condition, topN)
            elif action_type == "rerank":
                rec_condition, topN = action_params.split(',')[0].strip(), action_params.split(',')[1].strip()
                topN = int(topN)
                self.rerank_step(rec_condition, topN)
            elif action_type == "finish":
                answer = self.conclude_step(topK=self.final_length)
                self.answer = answer[1:-1]
                done = True
                self.obs = f"Episode finished, reward = {reward}\n"
        elif action.lower().startswith("finish"):
            # finish[] 형식이 아닌 경우도 처리
            answer = self.conclude_step(topK=self.final_length)
            self.answer = answer[1:-1]
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif re.search(r'think\[', action, re.IGNORECASE):
            self.obs = "Nice thought.  But an Action cannot be a think."
        else:
            self.obs = "Invalid action: {}. Expected format: retrieve[condition, topK], rerank[condition, topK], or finish[]".format(action)

        self.steps += 1

        return self.obs, reward, done, self._get_info()
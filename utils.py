import json
import pickle
import torch
from recbole.evaluator import Evaluator, Collector


dataset_name="mind"  # MIND 데이터셋 사용
test_version="test/"

backbone_model="SASRec"
file_list = ['XXX']  # saved text file name

model_file_dict = {
    'SASRec': {
        'ml-1m': {
            'None': 'SASRec-XXXX.pth',
            'genre': 'SASRec_AddInfo2-XXXX.pth',
            'release_year': 'SASRec_AddInfo2-XXXX.pth',
        },
        'amazon_book': {
            'None': 'SASRec-XXXX.pth',
            'price': 'SASRec_AddInfo2-XXXX.pth',
            'sales_rank': 'SASRec_AddInfo2-XXXX.pth',
        },
        'yelp': {
            'None': 'SASRec-XXXX.pth',
            'city': 'SASRec_AddInfo2-XXXX.pth',
            'stars': 'SASRec_AddInfo2-XXXX.pth',
            'categories': 'SASRec_AddInfo2-XXXX.pth',
        },
        'mind': {
            'None': 'SASRec-Nov-10-2025_16-08-05.pth',  # 기본 모델
            'category': 'SASRec_AddInfo2-Nov-10-2025_16-35-20.pth',  # 카테고리 기반 검색 도구
            'subcategory': 'SASRec_AddInfo2-Nov-10-2025_16-54-08.pth',  # 서브카테고리 기반 검색 도구
        }},
    'BERT4Rec': {
        'ml-1m': {
            'None': 'BERT4Rec-XXXX.pth',
            'genre': 'BERT4Rec_AddInfo-XXXX.pth',
            'release_year': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'amazon_book': {
            'None': 'BERT4Rec-XXXX.pth',
            'price': 'BERT4Rec_AddInfo-XXXX.pth',
            'sales_rank': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'yelp': {
            'None': 'BERT4Rec-XXXX.pth',
            'city': 'BERT4Rec_AddInfo-XXXX.pth',
            'stars': 'BERT4Rec_AddInfo-XXXX.pth',
            'categories': 'BERT4Rec_AddInfo-XXXX.pth',
        }
    }
}

model_BERT = {
    'SASRec': {
        'ml-1m': {
            'genre': 'SASRec_AddInfo-XXXX.pth',
        },
        'amazon_book': {
            'sales_rank': 'SASRec_AddInfo-XXXX.pth',
        },
        'yelp': {
            'categories': 'SASRec_AddInfo-XXXX.pth',
        }},
    'BERT4Rec': {
        'ml-1m': {
            'None': 'BERT4Rec-XXXX.pth',
            'genre': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'amazon_book': {
            'None': 'BERT4Rec-XXXX.pth',
            'sales_rank': 'BERT4Rec_AddInfo-XXXX.pth',
        },
        'yelp': {
            'None': 'BERT4Rec-XXXX.pth',
            'categories': 'BERT4Rec_AddInfo-XXXX.pth',
        }
    }
}


DATASET_ATT = model_file_dict[backbone_model][dataset_name].keys()

prompts_path = './dataset/prompts/' + test_version
# prompts_path5 = './dataset/prompts/length5/'
checkpoint_path = './dataset/saved_file/'

prompt_file = dataset_name + '_ICL.json'
profile = dataset_name + '_chat.pkl'
prompt_pattern = dataset_name + '_pattern.json'

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

# with open(prompts_path5 + profile, 'rb') as f:
#     uid_iid, user_profile, item_profile, itemID_name = pickle.load(f)

with open(prompts_path + profile, 'rb') as f:
    uid_iid, user_profile, item_profile, itemID_name = pickle.load(f)

with open(prompts_path + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

with open(prompts_path + prompt_pattern, 'r') as f:
    prompt_pattern = json.load(f)


token_path = prompts_path + dataset_name + '_ui_token.pkl'
with open(token_path, 'rb') as f:
    user_token_id, user_id_token, item_token_id, item_id_token = pickle.load(f)

# knowledge_prompt = prompt_dict['Reranking']


def extract_user_reclist(ranked_str):
    uid_topK = {}
    for uid, item_str in ranked_str.items():
        uid_topK[uid] = [item.split(', ')[0] for item in item_str[1:-1].strip().split('\n')]
    return uid_topK

def cleaning_user_itemList(ui_dict, topk=10):
    larger_than = 0
    smaller_than = 0
    uid_list = [u for u in ui_dict.keys()]
    for uid in uid_list:
        if len(ui_dict[uid]) == 10:
            continue
        elif len(ui_dict[uid]) > 10:
            larger_than += 1
            ui_dict[uid] = ui_dict[uid][:10]
        elif len(ui_dict[uid]) < 10:
            smaller_than += 1
            del ui_dict[uid]
    return ui_dict, larger_than, smaller_than

def check_itemList_length(ui_dict, topk=10):
    res_right = 1
    uid_list = [u for u in ui_dict.keys()]
    for uid in uid_list:
        if len(ui_dict[uid]) == 10:
            continue
        else:
            res_right = 0
    return res_right


def extract_and_check_cur_user_reclist(ranked_str, topk=10):
    """
    추천 리스트 형식 검증 함수
    Returns: 0 if valid, 1 if invalid
    """
    if not ranked_str or len(ranked_str) < 2:
        return 1  # invalid
    
    # 대괄호 제거
    ranked_str = ranked_str[1:-1] if ranked_str.startswith('[') and ranked_str.endswith(']') else ranked_str
    res_right = 1  # 1 = invalid, 0 = valid
    import re
    
    cur_user_reclist = []
    for item in ranked_str.strip().split('\n'):
        item = item.strip()
        if not item:
            continue
        
        # <ID> 형식에서 ID 추출 (여러 형식 지원)
        # 형식 1: <ID>, title, score
        # 형식 2: <ID> (줄바꿈으로만 구분)
        # 형식 3: <ID>
        id_match = re.search(r'<(\d+)>', item)
        if id_match:
            item_id = id_match.group(1)
            cur_user_reclist.append(item_id)
        else:
            # 기존 방식: 첫 번째 항목 추출
            first_part = item.split(',')[0].strip()  # 쉼표 하나로도 분리 시도
            # < > 제거
            first_part = first_part.strip('<').strip('>').strip()
            if first_part and first_part.isdigit():
                cur_user_reclist.append(first_part)
    
    # 길이 검증: 정확히 topk개여야 함
    if len(cur_user_reclist) != topk:
        return 1  # invalid: 길이가 맞지 않음
    
    # 아이템 ID 검증: 모든 ID가 item_token_id에 있어야 함
    for iid in cur_user_reclist:
        if not item_token_id.get(iid, 0):
            return 1  # invalid: 아이템 ID가 존재하지 않음
    
    return 0  # valid: 모든 검증 통과

def evaluate_user(user_id, pos_item, user_topK, user_num, item_num):
    topk_idx = torch.tensor(user_topK)
    positive_u = torch.tensor(user_id)     # minus 1 to ensure the matrix follows 0 -> user_num
    positive_i = torch.tensor(pos_item)

    # user_num = 943
    # item_num = 1683
    pos_matrix = torch.zeros((user_num, item_num), dtype=torch.int)
    pos_matrix[positive_u, positive_i] = 1
    pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
    pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
    result_matrix = torch.cat((pos_idx, pos_len_list), dim=1)
    data_struct = {}
    data_struct["rec.topk"] = result_matrix

    # Evaluate
    config = {}
    config["metric_decimal_place"] = 4
    config['metrics'] = ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
    config['topk'] = [1, 5, 10]  # HIT@1, MRR@5, nDCG@5, 그리고 기존 @10 메트릭 포함
    evaluator = Evaluator(config)

    result = evaluator.evaluate(data_struct)
    return result

def statstics_recedItems_LLM(uid_topK, top=10):
    item_freq = {}
    for u in uid_topK:
        for iid in uid_topK[u]:
            if item_freq.get(iid, 0):
                item_freq[iid] += 1
            else:
                item_freq[iid] = 1
    item_name_list = [ii for ii in item_freq.keys()]
    item_freq_list = [item_freq[ii] for ii in item_name_list]
    zipped_lists = zip(item_name_list, item_freq_list)
    # Sort the zipped lists based on the values in list2 in descending order
    sorted_lists = sorted(zipped_lists, key=lambda x: x[1], reverse=True)
    # Unzip the sorted lists
    sorted_list1, sorted_list2 = zip(*sorted_lists)
    # Retrieve the top 2 elements
    top_item_name = sorted_list1[:top]
    top_item_freq = sorted_list2[:top]
    return top_item_name, top_item_freq

def statstics_recedItems_CRS(uid_topK, top=10):
    item_freq = {}
    for i_list in uid_topK:
        for iid in i_list:
            if item_freq.get(iid, 0):
                item_freq[iid] += 1
            else:
                item_freq[iid] = 1
    item_name_list = [ii for ii in item_freq.keys()]
    item_freq_list = [item_freq[ii] for ii in item_name_list]
    zipped_lists = zip(item_name_list, item_freq_list)
    # Sort the zipped lists based on the values in list2 in descending order
    sorted_lists = sorted(zipped_lists, key=lambda x: x[1], reverse=True)
    # Unzip the sorted lists
    sorted_list1, sorted_list2 = zip(*sorted_lists)
    # Retrieve the top 2 elements
    top_item_name = sorted_list1[:top]
    top_item_freq = sorted_list2[:top]
    return top_item_name, top_item_freq
# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13, 2023/2/11
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import sys
from logging import getLogger

import pandas as pd


import pickle
from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

import random
from collections import defaultdict


def _tensor_col_to_list(col):
    if col is None:
        return []
    if hasattr(col, "cpu"):
        col = col.cpu().numpy()
    if hasattr(col, "tolist"):
        col = col.tolist()
    return list(col)


def _dump_labeled_benchmark_prompt_files(
    test_df,
    train_data,
    dataset_name,
    test_v,
    dataset_obj,
    his_len,
    sample_size,
    logger,
):
    """(user_id, timestamp) 당 여러 줄인 labeled 테스트 파일용 chat 프롬프트 피클 생성."""
    random.seed(2023)
    ds = dataset_obj
    if ds is None and train_data is not None and hasattr(train_data, "dataset"):
        ds = train_data.dataset
    if train_data is None or not hasattr(train_data, "dataset"):
        logger.error("labeled benchmark dump requires train_data")
        sys.exit(1)

    ti = train_data.dataset.inter_feat.interaction
    u_raw = _tensor_col_to_list(ti["user_id"])
    i_raw = _tensor_col_to_list(ti["item_id"])
    n = len(u_raw)
    try:
        r_raw = _tensor_col_to_list(ti["rating"])
    except (KeyError, TypeError):
        r_raw = []
    if len(r_raw) != n:
        r_raw = [1.0] * n
    try:
        ts_raw = _tensor_col_to_list(ti["timestamp"])
    except (KeyError, TypeError):
        ts_raw = []

    train_user_history = defaultdict(list)
    for uid_v, iid_v, rv, tv in zip(u_raw, i_raw, r_raw, ts_raw):
        if rv is not None and float(rv) < 0.5:
            continue
        try:
            if ds is not None:
                u_tok_res = ds.id2token(ds.uid_field, [uid_v])
                u_tok = u_tok_res[0] if u_tok_res else str(uid_v)
                if isinstance(u_tok, list):
                    u_tok = u_tok[0]
                i_tok_res = ds.id2token(ds.iid_field, [iid_v])
                i_tok = i_tok_res[0] if i_tok_res else str(iid_v)
                if isinstance(i_tok, list):
                    i_tok = i_tok[0]
            else:
                u_tok, i_tok = str(uid_v), str(iid_v)
        except Exception:
            u_tok, i_tok = str(uid_v), str(iid_v)
        train_user_history[str(u_tok)].append((str(i_tok), float(rv), float(tv)))

    uid_alias_map = {}
    uid_iid = {}
    uid_iid_his = {}
    uid_iid_hisScore = {}
    inst_counter = defaultdict(int)

    test_df = test_df.sort_values(["user_id", "timestamp"])
    for (_, _), g in test_df.groupby(["user_id", "timestamp"], sort=False):
        if len(g) < 2:
            continue
        uid_raw = g.iloc[0]["user_id"]
        base_uid = str(uid_raw)
        idx = inst_counter[base_uid]
        inst_counter[base_uid] += 1
        synthetic_uid = f"{base_uid}__inst{idx}"

        pos_rows = g[g["rating"] >= 0.5]
        if pos_rows.empty:
            continue
        pos_item_raw = pos_rows.iloc[0]["item_id"]
        try:
            if ds is not None:
                pi_res = ds.id2token(ds.iid_field, [pos_item_raw])
                pos_item = pi_res[0] if pi_res else str(pos_item_raw)
                if isinstance(pos_item, list):
                    pos_item = pos_item[0]
            else:
                pos_item = str(pos_item_raw)
        except Exception:
            pos_item = str(pos_item_raw)

        uid_iid[synthetic_uid] = str(pos_item)
        uid_alias_map[synthetic_uid] = base_uid

        hist = sorted(train_user_history.get(base_uid, []), key=lambda x: x[2])[
            -his_len:
        ]
        uid_iid_his[synthetic_uid] = [x[0] for x in hist]
        uid_iid_hisScore[synthetic_uid] = [x[1] for x in hist]

    users = list(uid_iid.keys())
    if not users:
        logger.error("labeled benchmark dump: no instances extracted from test file")
        sys.exit(1)

    actual_sample_size = min(sample_size, len(users))
    if actual_sample_size < len(users):
        sampled_users = random.sample(users, actual_sample_size)
        logger.info(
            f"Labeled benchmark: sampled {actual_sample_size} / {len(users)} instances"
        )
    else:
        sampled_users = users

    uid_iid_small = {u: uid_iid[u] for u in sampled_users}
    uid_iid_his_small = {u: uid_iid_his.get(u, []) for u in sampled_users}
    uid_iid_hisScore_small = {u: uid_iid_hisScore.get(u, []) for u in sampled_users}

    if test_v and not test_v.endswith("/"):
        test_v = test_v + "/"
    base_out = "./dataset/prompts/" + test_v

    file_path = base_out + dataset_name + "_uid_dict.pkl"
    with open(file_path, "wb") as f:
        pickle.dump((uid_iid_small, uid_iid_his_small, uid_iid_hisScore_small), f)

    alias_small = {
        k: uid_alias_map[k] for k in sampled_users if k in uid_alias_map
    }
    alias_path = base_out + dataset_name + "_uid_alias.pkl"
    with open(alias_path, "wb") as f:
        pickle.dump(alias_small, f)

    user_token_id = ds.field2token_id["user_id"]
    item_token_id = ds.field2token_id["item_id"]
    user_id_token = ds.field2id_token["user_id"]
    item_id_token = ds.field2id_token["item_id"]

    token_path = base_out + dataset_name + "_ui_token.pkl"
    with open(token_path, "wb") as f:
        pickle.dump((user_token_id, user_id_token, item_token_id, item_id_token), f)

    logger.info(f"Labeled benchmark: wrote {file_path}")
    logger.info(f"Labeled benchmark: wrote {alias_path}")
    logger.info(f"Labeled benchmark: wrote {token_path}")
    sys.exit(0)


def run_recbole(
    model=None, dataset=None, config_file_list=None, config_dict=None, saved=True
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if config['dump_to_chat']:
        try:
            sample_size = config['sample_size']
        except (KeyError, AttributeError):
            sample_size = 200  # 기본값 200
        dump_userInfo_chat(config['test_v'], config['dataset'], test_data, train_data, dataset, his_len=config['chat_hislen'], sample_size=sample_size)
        sys.exit()
    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

def dump_userInfo_chat(test_v, dataset_name, test_data, train_data=None, original_dataset=None, his_len=50, sample_size=200):
    logger = getLogger()
    uid_iid = {}
    uid_iid_his = {}
    uid_iid_hisScore = {}
    
    # 원본 test 파일을 직접 읽어서 모든 사용자 처리
    test_interaction = None
    dataset_obj = None
    
    # original_dataset에서 test 파일 직접 읽기
    if original_dataset is not None and hasattr(original_dataset, 'config'):
        try:
            benchmark_files = original_dataset.config['benchmark_filename']
            if benchmark_files and ('test' in benchmark_files or len(benchmark_files) > 1):
                import os
                try:
                    data_path = original_dataset.config['data_path']
                except (KeyError, AttributeError):
                    data_path = './dataset'
                
                # test 파일 경로 구성
                possible_paths = [
                    os.path.join(data_path, dataset_name, f"{dataset_name}.test.inter"),
                    os.path.join(data_path, f"{dataset_name}.test.inter"),
                    f"./dataset/{dataset_name}/{dataset_name}.test.inter",
                    f"./dataset/{dataset_name}.test.inter",
                ]
                
                test_file = None
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        test_file = abs_path
                        break
                
                if test_file and os.path.exists(test_file):
                    logger.info(f"Reading original test file directly: {test_file}")
                    import pandas as pd
                    test_df = pd.read_csv(test_file, sep='\t', header=0)
                    # 필드명에서 타입 제거 (예: 'user_id:token' -> 'user_id')
                    test_df.columns = [col.split(':')[0] for col in test_df.columns]
                    if dataset_obj is None:
                        if original_dataset is not None:
                            dataset_obj = original_dataset
                        elif hasattr(test_data, 'dataset'):
                            dataset_obj = test_data.dataset
                    # 동일 (user,timestamp) 에 여러 후보 줄이 있는 labeled 벤치마크 (예: mind_2000.test.inter)
                    if (
                        train_data is not None
                        and 'rating' in test_df.columns
                        and {'user_id', 'item_id', 'timestamp'}.issubset(
                            set(test_df.columns)
                        )
                    ):
                        grp_sz = test_df.groupby(['user_id', 'timestamp']).size()
                        if grp_sz.max() >= 2:
                            logger.info(
                                "Labeled benchmark test file detected "
                                f"(max {int(grp_sz.max())} rows per user,timestamp)."
                            )
                            _dump_labeled_benchmark_prompt_files(
                                test_df=test_df.copy(),
                                train_data=train_data,
                                dataset_name=dataset_name,
                                test_v=test_v,
                                dataset_obj=dataset_obj,
                                his_len=his_len,
                                sample_size=sample_size,
                                logger=logger,
                            )
                    test_interaction = test_df.to_dict('list')
                    logger.info(f"Read {len(test_interaction.get('user_id', []))} test interactions from original file")
                    # 원본 파일의 고유 사용자 수 확인
                    if 'user_id' in test_interaction:
                        original_unique_users = len(set(test_interaction['user_id']))
                        logger.info(f"Unique users in original test file: {original_unique_users}")
                    # dataset_obj 설정 (original_dataset 또는 test_data.dataset 사용)
                    if dataset_obj is None:
                        if original_dataset is not None:
                            dataset_obj = original_dataset
                        elif hasattr(test_data, 'dataset'):
                            dataset_obj = test_data.dataset
        except (KeyError, AttributeError) as e:
            logger.warning(f"Could not read original test file: {e}")
    
    # test_data에서 사용자별 테스트 아이템 추출 (원본 파일을 읽지 못한 경우에만)
    if test_interaction is None:
        try:
            # test_data의 구조 확인
            logger.info(f"test_data type: {type(test_data)}")
            logger.info(f"test_data.dataset type: {type(test_data.dataset)}")
            logger.info(f"test_data.dataset.inter_feat type: {type(test_data.dataset.inter_feat)}")
            
            # test_data가 DataLoader인 경우 dataset 속성 확인
            if hasattr(test_data, 'dataset'):
                dataset_obj = test_data.dataset
            else:
                dataset_obj = test_data
            
            # inter_feat 확인
            if hasattr(dataset_obj, 'inter_feat'):
                inter_feat = dataset_obj.inter_feat
                logger.info(f"inter_feat type: {type(inter_feat)}")
                if hasattr(inter_feat, 'interaction'):
                    test_interaction = inter_feat.interaction
                elif hasattr(inter_feat, '__dict__'):
                    test_interaction = inter_feat.__dict__
                else:
                    test_interaction = inter_feat
            else:
                logger.error("inter_feat not found in dataset")
                logger.info("Total users processed: 0")
                sys.exit(0)
            
            # test_interaction이 비어있으면 원본 데이터셋에서 직접 읽기
            if isinstance(test_interaction, dict) and len(test_interaction.get('user_id', [])) == 0:
                logger.info("test_interaction is empty, trying to read from original dataset")
                # original_dataset에서 test 데이터 직접 읽기
                if original_dataset is not None and hasattr(original_dataset, 'config'):
                    # benchmark_filename이 있으면 test 파일을 직접 읽어야 함
                    try:
                        benchmark_files = original_dataset.config['benchmark_filename']
                        if benchmark_files and ('test' in benchmark_files or len(benchmark_files) > 1):
                            # test 파일 경로 구성
                            import os
                            try:
                                data_path = original_dataset.config['data_path']
                            except (KeyError, AttributeError):
                                data_path = './dataset'
                            
                            # 여러 가능한 경로 시도
                            possible_paths = [
                                os.path.join(data_path, dataset_name, f"{dataset_name}.test.inter"),
                                os.path.join(data_path, f"{dataset_name}.test.inter"),
                                f"./dataset/{dataset_name}/{dataset_name}.test.inter",
                                f"./dataset/{dataset_name}.test.inter",
                            ]
                            
                            test_file = None
                            for path in possible_paths:
                                abs_path = os.path.abspath(path)
                                if os.path.exists(abs_path):
                                    test_file = abs_path
                                    break
                            
                            if test_file and os.path.exists(test_file):
                                logger.info(f"Reading test file directly: {test_file}")
                                import pandas as pd
                                test_df = pd.read_csv(test_file, sep='\t', header=0)
                                # 필드명에서 타입 제거 (예: 'user_id:token' -> 'user_id')
                                test_df.columns = [col.split(':')[0] for col in test_df.columns]
                                test_interaction = test_df.to_dict('list')
                                logger.info(f"Read {len(test_interaction.get('user_id', []))} test interactions from file")
                                # 원본 파일의 고유 사용자 수 확인
                                if 'user_id' in test_interaction:
                                    original_unique_users = len(set(test_interaction['user_id']))
                                    logger.info(f"Unique users in original test file: {original_unique_users}")
                            else:
                                logger.warning(f"Test file not found. Tried paths: {possible_paths}")
                    except (KeyError, AttributeError) as e:
                        logger.warning(f"Could not access benchmark_filename from config: {e}")
                
                # 여전히 비어있으면 inter_feat에서 직접 읽기
                if isinstance(test_interaction, dict) and len(test_interaction.get('user_id', [])) == 0:
                    logger.info("Still empty, trying to read from inter_feat directly")
                    # inter_feat가 DataFrame인 경우
                    if hasattr(inter_feat, 'to_dict'):
                        test_interaction = inter_feat.to_dict('list')
                    elif hasattr(inter_feat, '__dict__'):
                        test_interaction = {k: v for k, v in inter_feat.__dict__.items() if not k.startswith('_')}
                    # inter_feat 자체가 dict인 경우
                    elif isinstance(inter_feat, dict):
                        test_interaction = inter_feat
                    logger.info(f"After reading from inter_feat, user_id length: {len(test_interaction.get('user_id', []))}")
        except Exception as e:
            logger.error(f"Error accessing test_data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Total users processed: 0")
            sys.exit(0)
    
    # dataset_obj가 아직 설정되지 않았으면 test_data에서 가져오기
    if dataset_obj is None:
        if hasattr(test_data, 'dataset'):
            dataset_obj = test_data.dataset
        else:
            dataset_obj = test_data
    
    # test_interaction이 비어있는지 확인
    if test_interaction is None or (isinstance(test_interaction, dict) and len(test_interaction.get('user_id', [])) == 0):
        logger.error("test_interaction is empty or None")
        logger.info("Total users processed: 0")
        sys.exit(0)
    
    logger.info(f"Test interaction keys: {list(test_interaction.keys()) if isinstance(test_interaction, dict) else 'Not a dict'}")
    logger.info(f"Test interaction type: {type(test_interaction)}")
    
    # test_interaction이 텐서나 배열인 경우 처리
    if not isinstance(test_interaction, dict):
        logger.error(f"test_interaction is not a dict: {type(test_interaction)}")
        logger.info("Total users processed: 0")
        sys.exit(0)
    
    test_user_items = {}  # user_id -> [item_ids]
    
    # 원본 파일을 읽었는지 확인 (sequential 필드가 없으면 원본 파일)
    from_original_file = 'item_id_list' not in test_interaction
    
    # test_data가 sequential 필드를 가지고 있는지 확인
    has_sequential_fields = 'item_id_list' in test_interaction and 'item_length' in test_interaction
    logger.info(f"Reading from original file: {from_original_file}, has_sequential_fields: {has_sequential_fields}")
    
    if 'user_id' in test_interaction:
        user_ids = test_interaction['user_id']
        # 텐서나 배열을 리스트로 변환
        if hasattr(user_ids, 'cpu'):
            user_ids = user_ids.cpu().numpy()
        if hasattr(user_ids, 'tolist'):
            user_ids = user_ids.tolist()
        elif hasattr(user_ids, '__iter__') and not isinstance(user_ids, (str, bytes)):
            user_ids = list(user_ids)
        
        if hasattr(user_ids, '__len__'):
            logger.info(f"Number of test interactions: {len(user_ids)}")
            # 고유 사용자 수 확인
            try:
                if isinstance(user_ids, (list, tuple)):
                    unique_users_in_data = len(set(user_ids))
                elif hasattr(user_ids, 'tolist'):
                    unique_users_in_data = len(set(user_ids.tolist()))
                elif hasattr(user_ids, 'unique'):
                    unique_users_in_data = len(user_ids.unique())
                else:
                    unique_users_in_data = len(set(list(user_ids)))
                logger.info(f"Unique users in test_interaction (raw data): {unique_users_in_data}")
            except Exception as e:
                logger.warning(f"Could not count unique users: {e}")
        else:
            logger.info(f"user_ids type: {type(user_ids)}")
    else:
        logger.warning("'user_id' not found in test_interaction")
    
    if has_sequential_fields and not from_original_file:
        # sequential 필드가 있고 원본 파일이 아닌 경우 기존 방식 사용
        data = test_interaction
        processed_count = 0
        error_count = 0
        unique_user_ids = set()
        user_interaction_count = {}
        
        for (uid, iid, iid_his, i_len, iid_hisScore) in zip(data['user_id'], data['item_id'], data['item_id_list'], data['item_length'], data['rating_list']):
            try:
                # id2token이 2차원 리스트를 반환할 수 있으므로 평탄화
                u_token_result = test_data.dataset.id2token('user_id', [uid])
                if isinstance(u_token_result, list) and len(u_token_result) > 0:
                    u_token = u_token_result[0] if not isinstance(u_token_result[0], list) else u_token_result[0][0]
                else:
                    u_token = str(uid)
                
                unique_user_ids.add(u_token)
                user_interaction_count[u_token] = user_interaction_count.get(u_token, 0) + 1
                
                i_token_result = test_data.dataset.id2token('item_id', [iid])
                if isinstance(i_token_result, list) and len(i_token_result) > 0:
                    i_token = i_token_result[0] if not isinstance(i_token_result[0], list) else i_token_result[0][0]
                else:
                    i_token = str(iid)
                
                iid_his_token_result = test_data.dataset.id2token('item_id', iid_his)
                # 2차원 리스트를 1차원으로 평탄화
                if isinstance(iid_his_token_result, list):
                    iid_his_token = []
                    for item in iid_his_token_result:
                        if isinstance(item, list):
                            iid_his_token.extend(item)
                        else:
                            iid_his_token.append(item)
                else:
                    iid_his_token = iid_his_token_result

                # 같은 사용자가 여러 interaction을 가질 수 있으므로, 마지막 것만 저장 (또는 첫 번째 것만)
                # 기존 로직 유지: 마지막 interaction의 아이템을 저장
                uid_iid[u_token] = i_token
                if i_len >= his_len:
                    uid_iid_his[u_token] = iid_his_token[i_len - his_len:i_len]
                    uid_iid_hisScore[u_token] = iid_hisScore[i_len - his_len:i_len]
                else:
                    uid_iid_his[u_token] = iid_his_token[:i_len]
                    uid_iid_hisScore[u_token] = iid_hisScore[:i_len]
                processed_count += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing user {uid}, item {iid}: {e}")
                continue
        
        logger.info(f"Processed {processed_count} interactions successfully, {error_count} errors")
        logger.info(f"Total unique user IDs found: {len(unique_user_ids)}")
        logger.info(f"Unique users in uid_iid before sampling: {len(uid_iid)}")
        if len(user_interaction_count) > 0:
            interaction_counts = list(user_interaction_count.values())
            logger.info(f"Interactions per user - Min: {min(interaction_counts)}, Max: {max(interaction_counts)}, Avg: {sum(interaction_counts)/len(interaction_counts):.2f}")
            # 사용자당 interaction 수 분포 확인
            from collections import Counter
            count_distribution = Counter(interaction_counts)
            logger.info(f"Interaction count distribution (count: frequency): {dict(sorted(count_distribution.items())[:10])}")
    else:
        # sequential 필드가 없으면 train_data에서 히스토리 가져오기 (원본 파일을 읽은 경우 포함)
        logger.info("Sequential fields not found in test_data, using train_data for history")
        
        # test_data에서 사용자별 테스트 아이템 수집
        try:
            user_ids = test_interaction['user_id']
            item_ids = test_interaction['item_id']
            
            # 텐서나 배열을 리스트로 변환
            if hasattr(user_ids, 'cpu'):
                user_ids = user_ids.cpu().numpy()
            if hasattr(user_ids, 'tolist'):
                user_ids = user_ids.tolist()
            if hasattr(item_ids, 'cpu'):
                item_ids = item_ids.cpu().numpy()
            if hasattr(item_ids, 'tolist'):
                item_ids = item_ids.tolist()
            
            logger.info(f"Processing {len(user_ids)} test interactions")
            
            for uid, iid in zip(user_ids, item_ids):
                try:
                    # 원본 파일을 읽은 경우와 sequential 변환된 경우 구분
                    if from_original_file:
                        # 원본 파일을 읽은 경우, 문자열 그대로 사용하거나 dataset_obj에서 변환
                        if dataset_obj is not None:
                            try:
                                u_token_result = dataset_obj.id2token('user_id', [uid])
                                if isinstance(u_token_result, list) and len(u_token_result) > 0:
                                    u_token = u_token_result[0] if not isinstance(u_token_result[0], list) else u_token_result[0][0]
                                else:
                                    u_token = str(uid)
                            except:
                                u_token = str(uid)
                        else:
                            u_token = str(uid)
                        
                        if dataset_obj is not None:
                            try:
                                i_token_result = dataset_obj.id2token('item_id', [iid])
                                if isinstance(i_token_result, list) and len(i_token_result) > 0:
                                    i_token = i_token_result[0] if not isinstance(i_token_result[0], list) else i_token_result[0][0]
                                else:
                                    i_token = str(iid)
                            except:
                                i_token = str(iid)
                        else:
                            i_token = str(iid)
                    else:
                        # sequential 데이터인 경우
                        u_token_result = test_data.dataset.id2token('user_id', [uid])
                        if isinstance(u_token_result, list) and len(u_token_result) > 0:
                            u_token = u_token_result[0] if not isinstance(u_token_result[0], list) else u_token_result[0][0]
                        else:
                            u_token = str(uid)
                        
                        i_token_result = test_data.dataset.id2token('item_id', [iid])
                        if isinstance(i_token_result, list) and len(i_token_result) > 0:
                            i_token = i_token_result[0] if not isinstance(i_token_result[0], list) else i_token_result[0][0]
                        else:
                            i_token = str(iid)
                    
                    if u_token not in test_user_items:
                        test_user_items[u_token] = []
                    test_user_items[u_token].append(i_token)
                except Exception as e:
                    logger.warning(f"Error processing user {uid}, item {iid}: {e}")
                    continue
            
            logger.info(f"Found {len(test_user_items)} unique test users")
        except Exception as e:
            logger.error(f"Error processing test interactions: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # train_data에서 사용자별 히스토리 수집
        if train_data is not None:
            train_interaction = train_data.dataset.inter_feat.interaction
            if 'item_id_list' in train_interaction and 'item_length' in train_interaction:
                # train_data에 sequential 필드가 있으면 사용
                train_data_seq = train_interaction
                for (uid, iid_his, i_len, iid_hisScore) in zip(train_data_seq['user_id'], train_data_seq['item_id_list'], train_data_seq['item_length'], train_data_seq['rating_list']):
                    try:
                        # id2token이 2차원 리스트를 반환할 수 있으므로 평탄화
                        u_token_result = train_data.dataset.id2token('user_id', [uid])
                        if isinstance(u_token_result, list) and len(u_token_result) > 0:
                            u_token = u_token_result[0] if not isinstance(u_token_result[0], list) else u_token_result[0][0]
                        else:
                            u_token = str(uid)
                        
                        if u_token in test_user_items:
                            # 테스트 사용자인 경우에만 히스토리 저장
                            iid_his_token_result = train_data.dataset.id2token('item_id', iid_his)
                            # 2차원 리스트를 1차원으로 평탄화
                            if isinstance(iid_his_token_result, list):
                                iid_his_token = []
                                for item in iid_his_token_result:
                                    if isinstance(item, list):
                                        iid_his_token.extend(item)
                                    else:
                                        iid_his_token.append(item)
                            else:
                                iid_his_token = iid_his_token_result
                            
                            if i_len >= his_len:
                                uid_iid_his[u_token] = iid_his_token[i_len - his_len:i_len]
                                uid_iid_hisScore[u_token] = iid_hisScore[i_len - his_len:i_len]
                            else:
                                uid_iid_his[u_token] = iid_his_token[:i_len]
                                uid_iid_hisScore[u_token] = iid_hisScore[:i_len]
                    except Exception as e:
                        logger.warning(f"Error processing user {uid}, item {iid_his if isinstance(iid_his, (int, str)) else 'list'}: {e}")
                        continue
            else:
                # train_data에도 sequential 필드가 없으면 일반 interaction에서 히스토리 구성
                train_user_history = {}  # user_id -> [(item_id, rating, timestamp), ...]
                for uid, iid, rating, ts in zip(train_interaction['user_id'], train_interaction['item_id'], 
                                                 train_interaction.get('rating', [1.0]*len(train_interaction['user_id'])), 
                                                 train_interaction.get('timestamp', [0]*len(train_interaction['user_id']))):
                    u_token_result = train_data.dataset.id2token('user_id', [uid])
                    if isinstance(u_token_result, list) and len(u_token_result) > 0:
                        u_token = u_token_result[0] if not isinstance(u_token_result[0], list) else u_token_result[0][0]
                    else:
                        u_token = str(uid)
                    
                    if u_token not in train_user_history:
                        train_user_history[u_token] = []
                    
                    i_token_result = train_data.dataset.id2token('item_id', [iid])
                    if isinstance(i_token_result, list) and len(i_token_result) > 0:
                        i_token = i_token_result[0] if not isinstance(i_token_result[0], list) else i_token_result[0][0]
                    else:
                        i_token = str(iid)
                    
                    train_user_history[u_token].append((i_token, rating, ts))
                
                # 시간 순서대로 정렬하고 최근 his_len개만 사용
                for u_token in test_user_items.keys():
                    if u_token in train_user_history:
                        history = sorted(train_user_history[u_token], key=lambda x: x[2])[-his_len:]
                        uid_iid_his[u_token] = [item for item, _, _ in history]
                        uid_iid_hisScore[u_token] = [rating for _, rating, _ in history]
        
        # test_user_items에서 첫 번째 아이템을 테스트 아이템으로 사용
        for u_token, items in test_user_items.items():
            if items:
                uid_iid[u_token] = items[0]  # 첫 번째 아이템 사용
        
        logger.info(f"Created uid_iid for {len(uid_iid)} users")
        logger.info(f"Created uid_iid_his for {len(uid_iid_his)} users")
    
    users = list(uid_iid.keys())
    logger.info(f"Total unique users before sampling: {len(users)}")
    logger.info(f"Sample size requested: {sample_size}")
    # 사용자 수가 sample_size보다 적으면 전체 사용자 사용, 그렇지 않으면 sample_size명 샘플링
    actual_sample_size = min(sample_size, len(users))
    if actual_sample_size < len(users):
        sampled_users = random.sample(users, actual_sample_size)
        logger.info(f"Sampling {actual_sample_size} users from {len(users)} total users")
    else:
        sampled_users = users
        logger.info(f"Using all {len(users)} users (sample_size >= total users)")
    uid_iid_small = {u: uid_iid[u] for u in sampled_users}
    # uid_iid_his에 없는 사용자는 빈 리스트로 처리
    uid_iid_his_small = {u: uid_iid_his.get(u, []) for u in sampled_users}
    uid_iid_hisScore_small = {u: uid_iid_hisScore.get(u, []) for u in sampled_users}

    # file_path = './dataset/prompts/' + dataset_name + '_uid_dict.pkl'
    if test_v and not test_v.endswith('/'):
        test_v = test_v + '/'
    file_path = './dataset/prompts/' + test_v + dataset_name + '_uid_dict.pkl'
    # with open(file_path, 'wb') as f:
    #     pickle.dump((uid_iid, uid_iid_his, uid_iid_hisScore), f)
    with open(file_path, 'wb') as f:
        pickle.dump((uid_iid_small, uid_iid_his_small, uid_iid_hisScore_small), f)
    user_token_id = test_data.dataset.field2token_id['user_id']
    item_token_id = test_data.dataset.field2token_id['item_id']
    user_id_token = test_data.dataset.field2id_token['user_id']
    item_id_token = test_data.dataset.field2id_token['item_id']

    token_path = './dataset/prompts/' + test_v + dataset_name + '_ui_token.pkl'
    with open(token_path, 'wb') as f:
        pickle.dump((user_token_id, user_id_token, item_token_id, item_id_token), f)
    
    logger = getLogger()
    logger.info(f"Successfully created {file_path}")
    logger.info(f"Successfully created {token_path}")
    logger.info(f"Total users processed: {len(sampled_users)}")
    sys.exit(0)
    




def run_recboles(rank, *args):
    ip, port, world_size, nproc, offset = args[3:]
    args = args[:3]
    run_recbole(
        *args,
        config_dict={
            "local_rank": rank,
            "world_size": world_size,
            "ip": ip,
            "port": port,
            "nproc": nproc,
            "offset": offset,
        },
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def save_split_dataset(dataset, split=None, topK=1):
    '''
    Top K means remain the most or second most attribute for each user.
    ...... to construct the 
    '''
    if split == 'class':
        item_df = dataset.item_feat
        item_df['class'] = item_df['class'].apply(lambda x: ','.join(str(x)))
    else:
        item_df = dataset.item_feat
    group_df = dataset.inter_feat.groupby(dataset.uid_field)
    

    group_list = list(group_df)
    for uid, udf in group_list:
        udf = pd.merge(udf, item_df, on=dataset.iid_field)
        udf_split_topK = udf.groupby(split)[split].count().sort_values().index[-topK:].values
        # write dataframe rows into files.  which in udf_split_topK.
        for index, row in udf.iterrows():
            if row[split] in udf_split_topK:
                pass


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file, weights_only=False)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    # strict=False를 사용하여 일치하지 않는 레이어를 건너뛰도록 함
    # (아이템 수가 다른 경우에도 로드 가능)
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.warning(f"모델 로딩 시 size mismatch 발생. size mismatch가 있는 키를 필터링합니다: {str(e)}")
            # size mismatch가 있는 키를 필터링
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        logger.warning(f"키 '{key}'의 shape가 일치하지 않습니다. checkpoint: {value.shape}, model: {model_state_dict[key].shape}. 건너뜁니다.")
                else:
                    logger.warning(f"키 '{key}'가 모델에 없습니다. 건너뜁니다.")
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            raise
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data

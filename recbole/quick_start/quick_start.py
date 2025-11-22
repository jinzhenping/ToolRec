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
        dump_userInfo_chat(config['test_v'], config['dataset'], test_data, train_data, dataset, his_len=config['chat_hislen'])
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

def dump_userInfo_chat(test_v, dataset_name, test_data, train_data=None, original_dataset=None, his_len=10):
    logger = getLogger()
    uid_iid = {}
    uid_iid_his = {}
    uid_iid_hisScore = {}
    
    # test_data에서 사용자별 테스트 아이템 추출
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
        
        logger.info(f"Test interaction keys: {list(test_interaction.keys()) if isinstance(test_interaction, dict) else 'Not a dict'}")
        logger.info(f"Test interaction type: {type(test_interaction)}")
        
        # test_interaction이 텐서나 배열인 경우 처리
        if not isinstance(test_interaction, dict):
            logger.error(f"test_interaction is not a dict: {type(test_interaction)}")
            logger.info("Total users processed: 0")
            sys.exit(0)
        
        test_user_items = {}  # user_id -> [item_ids]
        
        # test_data가 sequential 필드를 가지고 있는지 확인
        has_sequential_fields = 'item_id_list' in test_interaction and 'item_length' in test_interaction
        logger.info(f"Has sequential fields: {has_sequential_fields}")
        
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
            else:
                logger.info(f"user_ids type: {type(user_ids)}")
        else:
            logger.warning("'user_id' not found in test_interaction")
    except Exception as e:
        logger.error(f"Error accessing test_interaction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Total users processed: 0")
        sys.exit(0)
    
    if has_sequential_fields:
        # sequential 필드가 있으면 기존 방식 사용
        data = test_interaction
        for (uid, iid, iid_his, i_len, iid_hisScore) in zip(data['user_id'], data['item_id'], data['item_id_list'], data['item_length'], data['rating_list']):
            u_token = test_data.dataset.id2token('user_id', [uid])[0]
            i_token = test_data.dataset.id2token('item_id', [iid])[0]
            iid_his_token = test_data.dataset.id2token('item_id', iid_his)

            uid_iid[u_token] = i_token
            if i_len >= his_len:
                uid_iid_his[u_token] = iid_his_token[i_len - his_len:i_len]
                uid_iid_hisScore[u_token] = iid_hisScore[i_len - his_len:i_len]
            else:
                uid_iid_his[u_token] = iid_his_token[:i_len]
                uid_iid_hisScore[u_token] = iid_hisScore[:i_len]
    else:
        # sequential 필드가 없으면 train_data에서 히스토리 가져오기
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
                    u_token = test_data.dataset.id2token('user_id', [uid])[0]
                    i_token = test_data.dataset.id2token('item_id', [iid])[0]
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
                    u_token = train_data.dataset.id2token('user_id', [uid])[0]
                    if u_token in test_user_items:
                        # 테스트 사용자인 경우에만 히스토리 저장
                        iid_his_token = train_data.dataset.id2token('item_id', iid_his)
                        if i_len >= his_len:
                            uid_iid_his[u_token] = iid_his_token[i_len - his_len:i_len]
                            uid_iid_hisScore[u_token] = iid_hisScore[i_len - his_len:i_len]
                        else:
                            uid_iid_his[u_token] = iid_his_token[:i_len]
                            uid_iid_hisScore[u_token] = iid_hisScore[:i_len]
            else:
                # train_data에도 sequential 필드가 없으면 일반 interaction에서 히스토리 구성
                train_user_history = {}  # user_id -> [(item_id, rating, timestamp), ...]
                for uid, iid, rating, ts in zip(train_interaction['user_id'], train_interaction['item_id'], 
                                                 train_interaction.get('rating', [1.0]*len(train_interaction['user_id'])), 
                                                 train_interaction.get('timestamp', [0]*len(train_interaction['user_id']))):
                    u_token = train_data.dataset.id2token('user_id', [uid])[0]
                    if u_token not in train_user_history:
                        train_user_history[u_token] = []
                    i_token = train_data.dataset.id2token('item_id', [iid])[0]
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
    # 사용자 수가 200명보다 적으면 전체 사용자 사용, 그렇지 않으면 200명 샘플링
    sample_size = min(200, len(users))
    if sample_size < len(users):
        sampled_users = random.sample(users, sample_size)
    else:
        sampled_users = users
    uid_iid_small = {u: uid_iid[u] for u in sampled_users}
    uid_iid_his_small = {u: uid_iid_his[u] for u in sampled_users}
    uid_iid_hisScore_small = {u: uid_iid_hisScore[u] for u in sampled_users}

    # file_path = './dataset/prompts/' + dataset + '_uid_dict.pkl'
    if test_v and not test_v.endswith('/'):
        test_v = test_v + '/'
    file_path = './dataset/prompts/' + test_v + dataset + '_uid_dict.pkl'
    # with open(file_path, 'wb') as f:
    #     pickle.dump((uid_iid, uid_iid_his, uid_iid_hisScore), f)
    with open(file_path, 'wb') as f:
        pickle.dump((uid_iid_small, uid_iid_his_small, uid_iid_hisScore_small), f)
    user_token_id = test_data.dataset.field2token_id['user_id']
    item_token_id = test_data.dataset.field2token_id['item_id']
    user_id_token = test_data.dataset.field2id_token['user_id']
    item_id_token = test_data.dataset.field2id_token['item_id']

    token_path = './dataset/prompts/' + test_v + dataset + '_ui_token.pkl'
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
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data

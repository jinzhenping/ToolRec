"""
MIND 데이터셋을 RecBole 형식으로 변환하는 스크립트
"""
import pandas as pd
import os
import pickle
import random
from collections import defaultdict

def convert_mind_to_recbole():
    """
    MIND 데이터셋을 RecBole 형식으로 변환
    """
    # 파일 경로 설정
    news_file = "MIND_news.tsv"
    users_file = "MIND_test_(1000)_unique_users.tsv"
    output_dir = "dataset/mind"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. 뉴스 정보 파일 읽기...")
    # 뉴스 정보 읽기 (컬럼: news_id, category, subcategory, title, abstract)
    news_df = pd.read_csv(news_file, sep='\t', header=None, 
                          names=['news_id', 'category', 'subcategory', 'title', 'abstract'],
                          encoding='utf-8')
    
    print(f"   - 총 {len(news_df)}개의 뉴스 로드됨")
    
    # 뉴스 ID를 정수로 변환 (N1 -> 1, N2 -> 2 등)
    news_df['news_id_clean'] = news_df['news_id'].str.replace('N', '').astype(int)
    news_id_mapping = dict(zip(news_df['news_id'], news_df['news_id_clean']))
    
    print("2. 사용자 데이터 읽기...")
    # 사용자 데이터 읽기
    users_data = []
    with open(users_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                user_id = parts[0]
                history = parts[1].split()  # 읽은 뉴스 시퀀스
                test_items = parts[2].split()  # 첫번째가 groundtruth, 나머지가 negative
                
                users_data.append({
                    'user_id': user_id,
                    'history': history,
                    'test_items': test_items
                })
    
    print(f"   - 총 {len(users_data)}명의 사용자 로드됨")
    
    print("3. 테스트 데이터 저장...")
    # 테스트 데이터 저장 (groundtruth + negative 샘플)
    # 순서 편향을 방지하기 위해 각 사용자별로 랜덤하게 섞음
    test_data = []
    test_groundtruth = {}  # 사용자별 groundtruth 저장
    test_item_positions = {}  # 사용자별 groundtruth의 위치 저장 (평가 시 사용)
    
    # 재현 가능성을 위해 시드 설정
    random.seed(42)
    
    for user_data in users_data:
        user_id = user_data['user_id']
        test_items = user_data['test_items']
        
        if len(test_items) > 0:
            # 첫 번째가 groundtruth
            groundtruth_id = test_items[0]
            if groundtruth_id in news_id_mapping:
                groundtruth_item_id = news_id_mapping[groundtruth_id]
                test_groundtruth[user_id] = groundtruth_item_id
                
                # 모든 아이템(groundtruth + negative)을 리스트로 만들기
                user_test_items = []
                
                # groundtruth를 positive로 추가
                user_test_items.append({
                    'item_id': groundtruth_item_id,
                    'rating': 1.0,
                    'is_groundtruth': True
                })
                
                # negative 샘플들 추가
                for neg_item_id in test_items[1:]:
                    if neg_item_id in news_id_mapping:
                        neg_item_id_clean = news_id_mapping[neg_item_id]
                        user_test_items.append({
                            'item_id': neg_item_id_clean,
                            'rating': 0.0,
                            'is_groundtruth': False
                        })
                
                # 순서 편향 방지를 위해 랜덤하게 섞기
                random.shuffle(user_test_items)
                
                # groundtruth의 위치 찾기
                groundtruth_position = None
                for idx, item in enumerate(user_test_items):
                    if item['is_groundtruth']:
                        groundtruth_position = idx
                        break
                
                test_item_positions[user_id] = groundtruth_position
                
                # 섞인 순서로 테스트 데이터에 추가
                for item in user_test_items:
                    test_data.append({
                        'user_id': user_id,
                        'item_id': item['item_id'],
                        'rating': item['rating'],
                        'timestamp': 2000000000  # 테스트 데이터는 더 큰 타임스탬프 사용
                    })
    
    print(f"   - 총 {len(test_groundtruth)}명의 사용자에 대한 테스트 데이터 생성")
    print(f"   - 총 {len(test_data)}개의 테스트 상호작용 (groundtruth + negative)")
    
    # 테스트 데이터를 pickle로 저장 (나중에 평가 시 사용)
    test_gt_path = os.path.join(output_dir, "mind_test_groundtruth.pkl")
    with open(test_gt_path, 'wb') as f:
        pickle.dump({
            'groundtruth': test_groundtruth,  # 사용자별 groundtruth 아이템 ID
            'positions': test_item_positions  # 사용자별 groundtruth의 위치 (섞인 후)
        }, f)
    print(f"   - {test_gt_path} 저장 완료 (groundtruth + positions)")
    
    # 테스트 상호작용을 별도 .inter 파일로 저장
    test_inter_output = os.path.join(output_dir, "mind.test.inter")
    with open(test_inter_output, 'w', encoding='utf-8') as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for test_item in test_data:
            f.write(f"{test_item['user_id']}\t{test_item['item_id']}\t{test_item['rating']}\t{test_item['timestamp']}\n")
    print(f"   - {test_inter_output} 저장 완료")
    
    print("4. 학습용 상호작용 데이터(.inter) 생성...")
    # 상호작용 데이터 생성
    interactions = []
    timestamp = 1000000000  # 시작 타임스탬프 (UNIX timestamp 형식)
    missing_news = set()
    
    for user_data in users_data:
        user_id = user_data['user_id']
        history = user_data['history']
        
        # 각 사용자의 읽은 뉴스 시퀀스를 시간 순서대로 기록
        for news_id in history:
            if news_id in news_id_mapping:
                item_id = news_id_mapping[news_id]
                interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': 1.0,  # 읽었다는 의미로 1.0
                    'timestamp': timestamp
                })
                timestamp += 1
            else:
                missing_news.add(news_id)
    
    if missing_news:
        print(f"   - 경고: {len(missing_news)}개의 뉴스 ID가 뉴스 파일에 없습니다 (상호작용에서 제외됨)")
        print(f"   - 예시: {list(missing_news)[:5]}")
    
    inter_df = pd.DataFrame(interactions)
    print(f"   - 총 {len(inter_df)}개의 상호작용 생성됨")
    
    # RecBole 형식으로 저장 (.inter 파일)
    inter_output = os.path.join(output_dir, "mind.inter")
    with open(inter_output, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        # 데이터 작성
        for _, row in inter_df.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\t{row['rating']}\t{row['timestamp']}\n")
    
    print(f"   - {inter_output} 저장 완료")
    
    print("5. 아이템 정보 파일(.item) 생성...")
    # 아이템 정보 파일 생성
    # title을 token_seq로 처리 (공백으로 구분된 단어들)
    item_output = os.path.join(output_dir, "mind.item")
    with open(item_output, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("item_id:token\ttitle:token_seq\tcategory:token\tsubcategory:token\n")
        # 데이터 작성
        for _, row in news_df.iterrows():
            item_id = row['news_id_clean']
            title = str(row['title']).replace('\t', ' ').replace('\n', ' ')  # 탭과 줄바꿈 제거
            category = str(row['category'])
            subcategory = str(row['subcategory'])
            
            f.write(f"{item_id}\t{title}\t{category}\t{subcategory}\n")
    
    print(f"   - {item_output} 저장 완료")
    
    print("6. 사용자 정보 파일(.user) 생성...")
    # 사용자 정보 파일 생성 (user_id만 포함)
    user_ids = set([user_data['user_id'] for user_data in users_data])
    user_output = os.path.join(output_dir, "mind.user")
    with open(user_output, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("user_id:token\n")
        # 데이터 작성
        for user_id in sorted(user_ids, key=lambda x: int(x)):
            f.write(f"{user_id}\n")
    
    print(f"   - {user_output} 저장 완료")
    
    print("\n7. 데이터셋 통계 정보:")
    print(f"   - 사용자 수: {len(user_ids)}")
    print(f"   - 아이템 수: {len(news_df)}")
    print(f"   - 상호작용 수: {len(inter_df)}")
    print(f"   - 평균 사용자당 상호작용 수: {len(inter_df) / len(user_ids):.2f}")
    
    # 사용자별 상호작용 수 확인
    user_inter_counts = inter_df.groupby('user_id').size()
    print(f"   - 최소 상호작용 수: {user_inter_counts.min()}")
    print(f"   - 최대 상호작용 수: {user_inter_counts.max()}")
    print(f"   - 중간값 상호작용 수: {user_inter_counts.median():.2f}")
    
    print("\n변환 완료!")
    print(f"\n생성된 파일:")
    print(f"  - {inter_output} (학습용 상호작용)")
    print(f"  - {test_inter_output} (테스트용 상호작용)")
    print(f"  - {test_gt_path} (테스트 groundtruth)")
    print(f"  - {item_output}")
    print(f"  - {user_output}")
    
    # README 파일 생성
    readme_content = """INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file mind.inter comprising the interactions of users over the news.
Each record/line in the file has the following fields: user_id, item_id, rating, timestamp

user_id: the id of the users and its type is token. 
item_id: the id of the news and its type is token.
rating: the rating of the users over the news (always 1.0 for read news), and its type is float.
timestamp: the timestamp of the interaction, and its type is float.

NEWS INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file mind.item comprising the attributes of the news.
Each record/line in the file has the following fields: item_id, title, category, subcategory
 
item_id: the id of the news and its type is token.
title: the title of the news, and its type is token_seq.
category: the category of the news, and its type is token.
subcategory: the subcategory of the news, and its type is token.


USERS INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file mind.user comprising the user IDs.
Each record/line in the file has the following fields: user_id
 
user_id: the id of the users and its type is token.
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"  - {readme_path}")

if __name__ == "__main__":
    convert_mind_to_recbole()


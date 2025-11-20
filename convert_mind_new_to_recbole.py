"""
mind_new 디렉토리의 데이터를 RecBole 형식으로 변환하는 스크립트
- behaviors_new.tsv: 사용자별 읽은 뉴스 시퀀스
- news.tsv: 뉴스 정보 (news_id, category, subcategory, title, ...)
"""
import pandas as pd
import os
from collections import defaultdict

def convert_mind_new_to_recbole():
    """
    mind_new 디렉토리의 데이터를 RecBole 형식으로 변환
    """
    # 파일 경로 설정
    news_file = "mind_new/news.tsv"
    behaviors_file = "mind_new/behaviors_new.tsv"
    output_dir = "dataset/mind"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. 뉴스 정보 파일 읽기...")
    # 뉴스 정보 읽기
    # 컬럼: news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
    news_df = pd.read_csv(news_file, sep='\t', header=None, 
                          names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
                                 'url', 'title_entities', 'abstract_entities'],
                          encoding='utf-8', on_bad_lines='skip')
    
    print(f"   - 총 {len(news_df)}개의 뉴스 로드됨")
    
    # 뉴스 ID를 정수로 변환 (N1 -> 1, N2 -> 2 등)
    news_df['news_id_clean'] = news_df['news_id'].str.replace('N', '').astype(int)
    news_id_mapping = dict(zip(news_df['news_id'], news_df['news_id_clean']))
    print(f"   - 뉴스 ID 매핑 완료 (예: N88753 -> {news_id_mapping.get('N88753', 'N/A')})")
    
    print("2. 사용자 행동 데이터 읽기...")
    # behaviors_new.tsv 읽기
    # 형식: user_id \t news_id1 news_id2 news_id3 ...
    interactions = []
    user_ids = set()
    timestamp = 1000000000  # 시작 타임스탬프
    missing_news = set()
    total_interactions = 0
    
    with open(behaviors_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            user_id = parts[0]
            news_ids = parts[1].split()  # 공백으로 구분된 뉴스 ID 리스트
            
            user_ids.add(user_id)
            
            # 각 사용자의 읽은 뉴스 시퀀스를 시간 순서대로 기록
            for news_id in news_ids:
                if news_id in news_id_mapping:
                    item_id = news_id_mapping[news_id]
                    interactions.append({
                        'user_id': user_id,
                        'item_id': item_id,
                        'rating': 1.0,  # 읽었다는 의미로 1.0
                        'timestamp': timestamp
                    })
                    timestamp += 1
                    total_interactions += 1
                else:
                    missing_news.add(news_id)
            
            if line_num % 10000 == 0:
                print(f"   - {line_num}줄 처리 중... (현재 {total_interactions}개 상호작용)")
    
    if missing_news:
        print(f"   - 경고: {len(missing_news)}개의 뉴스 ID가 뉴스 파일에 없습니다 (상호작용에서 제외됨)")
        print(f"   - 예시: {list(missing_news)[:5]}")
    
    print(f"   - 총 {len(user_ids)}명의 사용자")
    print(f"   - 총 {len(interactions)}개의 상호작용 생성됨")
    
    print("3. 학습용 상호작용 데이터(.inter) 생성...")
    inter_df = pd.DataFrame(interactions)
    
    # RecBole 형식으로 저장 (.inter 파일)
    inter_output = os.path.join(output_dir, "mind.inter")
    with open(inter_output, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        # 데이터 작성
        for _, row in inter_df.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\t{row['rating']}\t{row['timestamp']}\n")
    
    print(f"   - {inter_output} 저장 완료")
    
    # mind.train.inter도 생성 (기존 설정 파일이 benchmark_filename을 사용하므로)
    train_inter_output = os.path.join(output_dir, "mind.train.inter")
    with open(train_inter_output, 'w', encoding='utf-8') as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for _, row in inter_df.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\t{row['rating']}\t{row['timestamp']}\n")
    
    print(f"   - {train_inter_output} 저장 완료 (benchmark_filename용)")
    
    # mind.test.inter도 생성 (빈 파일 또는 기존 파일 유지)
    # 학습만 하고 싶다면 빈 파일을 생성하거나, 기존 파일을 유지
    test_inter_output = os.path.join(output_dir, "mind.test.inter")
    if not os.path.exists(test_inter_output):
        # 빈 테스트 파일 생성 (헤더만)
        with open(test_inter_output, 'w', encoding='utf-8') as f:
            f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        print(f"   - {test_inter_output} 생성 완료 (빈 파일, 나중에 테스트 데이터 추가 가능)")
    else:
        print(f"   - {test_inter_output} 이미 존재 (기존 파일 유지)")
    
    print("4. 아이템 정보 파일(.item) 생성...")
    # 아이템 정보 파일 생성
    item_output = os.path.join(output_dir, "mind.item")
    with open(item_output, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("item_id:token\ttitle:token_seq\tcategory:token\tsubcategory:token\n")
        # 데이터 작성
        for _, row in news_df.iterrows():
            item_id = row['news_id_clean']
            title = str(row['title']).replace('\t', ' ').replace('\n', ' ')  # 탭과 줄바꿈 제거
            category = str(row['category']) if pd.notna(row['category']) else 'unknown'
            subcategory = str(row['subcategory']) if pd.notna(row['subcategory']) else 'unknown'
            
            f.write(f"{item_id}\t{title}\t{category}\t{subcategory}\n")
    
    print(f"   - {item_output} 저장 완료")
    
    print("5. 사용자 정보 파일(.user) 생성...")
    # 사용자 정보 파일 생성 (user_id만 포함)
    user_output = os.path.join(output_dir, "mind.user")
    with open(user_output, 'w', encoding='utf-8') as f:
        # 헤더 작성
        f.write("user_id:token\n")
        # 데이터 작성 (정수로 정렬)
        for user_id in sorted(user_ids, key=lambda x: int(x)):
            f.write(f"{user_id}\n")
    
    print(f"   - {user_output} 저장 완료")
    
    print("\n6. 데이터셋 통계 정보:")
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
    print(f"  - {train_inter_output} (학습용 상호작용 - benchmark_filename용)")
    print(f"  - {item_output}")
    print(f"  - {user_output}")
    print(f"\n다음 명령어로 학습을 시작할 수 있습니다:")
    print(f"  python run_recbole.py --model=SASRec --dataset=mind --config_files=dataset/mind/mind.yaml")

if __name__ == "__main__":
    convert_mind_new_to_recbole()


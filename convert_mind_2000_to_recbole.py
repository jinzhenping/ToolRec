"""
MIND_2000 폴더의 TSV를 RecBole atomic 형식으로 변환합니다.

입력 (기본 경로: ./MIND_2000/)
  - MIND_train.tsv:    user, clicked_news, candidate_news, clicked (공백 구분 라벨)
  - MIND_validation.tsv / MIND_test.tsv: user, clicked_news, candidate_news (라벨 없음)
  - MIND_news.tsv:     news_id, category, subcategory, title, (본문…)

라벨이 없는 val/test 행은 학습 TSV와 동일하게 \"첫 번째 후보 = 정답(1), 나머지 0\"으로 처리합니다
(학습 파일의 clicked가 항상 1 0 0 0 0 형태인 것과 동일한 가정).

출력: ./dataset/mind_2000/
  mind_2000.train.inter, mind_2000.valid.inter, mind_2000.test.inter
  mind_2000.item, mind_2000.user
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from collections import defaultdict

import pandas as pd

INTER_HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
ITEM_HEADER = "item_id:token\ttitle:token_seq\tcategory:token\tsubcategory:token\n"
USER_HEADER = "user_id:token\n"


def _clean_cell(s: str) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").strip()


def load_news(news_path: str) -> tuple[dict[str, dict[str, str]], set[str]]:
    """news_id -> {category, subcategory, title}; 전체 news id 집합."""
    df = pd.read_csv(news_path, sep="\t", header=None, on_bad_lines="skip")
    if df.shape[1] < 4:
        raise ValueError(f"뉴스 파일 컬럼이 부족합니다: {news_path} (최소 4열 필요)")
    df = df.iloc[:, :5].copy()
    df.columns = ["item_id", "category", "subcategory", "title", "body"]
    info: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        nid = str(row["item_id"]).strip()
        if not nid:
            continue
        info[nid] = {
            "category": _clean_cell(row["category"]) if pd.notna(row["category"]) else "unknown",
            "subcategory": _clean_cell(row["subcategory"]) if pd.notna(row["subcategory"]) else "unknown",
            "title": _clean_cell(row["title"]) if pd.notna(row["title"]) else "unknown",
        }
    return info, set(info.keys())


def parse_behavior_line(line: str) -> tuple[str, list[str], list[str], list[int] | None]:
    """한 줄 -> user, history_ids, candidate_ids, labels 또는 None."""
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    if line.lower().startswith("user\t"):
        raise ValueError("header")
    parts = line.split("\t", 3)
    if len(parts) < 3:
        raise ValueError(f"bad columns: {line[:80]}")
    uid = parts[0].strip()
    hist = [x for x in parts[1].split() if x]
    cands = [x for x in parts[2].split() if x]
    labels: list[int] | None = None
    if len(parts) == 4:
        labels = [int(float(x)) for x in parts[3].split() if x]
    return uid, hist, cands, labels


def labels_from_candidates(
    cands: list[str], labels: list[int] | None
) -> list[float]:
    if labels is not None:
        if len(labels) != len(cands):
            raise ValueError(f"후보 수({len(cands)})와 라벨 수({len(labels)}) 불일치")
        return [float(x) for x in labels]
    # val/test: 첫 후보만 positive (train TSV 패턴과 동일)
    return [1.0] + [0.0] * (len(cands) - 1)


def ground_truth_item(cands: list[str], ratings: list[float]) -> str:
    """rating>=0.5 인 첫 후보를 정답으로 사용. 없으면 첫 후보."""
    for c, r in zip(cands, ratings):
        if r >= 0.5:
            return c
    return cands[0]


def convert(
    input_dir: str,
    output_dir: str,
    shuffle_seed: int = 2023,
) -> None:
    news_path = os.path.join(input_dir, "MIND_news.tsv")
    train_path = os.path.join(input_dir, "MIND_train.tsv")
    valid_path = os.path.join(input_dir, "MIND_validation.tsv")
    test_path = os.path.join(input_dir, "MIND_test.tsv")

    for p in (news_path, train_path, valid_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    os.makedirs(output_dir, exist_ok=True)
    news_info, news_ids = load_news(news_path)

    all_users: set[str] = set()
    missing_news: defaultdict[str, int] = defaultdict(int)

    # ---------- item ----------
    item_path = os.path.join(output_dir, "mind_2000.item")
    with open(item_path, "w", encoding="utf-8") as f:
        f.write(ITEM_HEADER)
        for nid in sorted(news_info.keys(), key=lambda x: (len(x), x)):
            meta = news_info[nid]
            f.write(
                f"{nid}\t{meta['title']}\t{meta['category']}\t{meta['subcategory']}\n"
            )

    # ---------- train: 순차 학습용 positive만 (history + 라벨이 1인 후보) ----------
    train_out = os.path.join(output_dir, "mind_2000.train.inter")
    user_ts: defaultdict[str, int] = defaultdict(int)
    with open(train_path, "r", encoding="utf-8") as fin, open(
        train_out, "w", encoding="utf-8"
    ) as fout:
        fout.write(INTER_HEADER)
        first = True
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            if first and raw.lower().startswith("user\t"):
                first = False
                continue
            first = False
            try:
                uid, hist, cands, labels = parse_behavior_line(line)
            except ValueError:
                continue
            all_users.add(uid)
            ts = user_ts[uid]
            for nid in hist:
                if nid not in news_ids:
                    missing_news[nid] += 1
                    continue
                fout.write(f"{uid}\t{nid}\t1.0\t{ts}\n")
                ts += 1
            rs = labels_from_candidates(cands, labels)
            for nid, r in zip(cands, rs):
                if r < 0.5:
                    continue
                if nid not in news_ids:
                    missing_news[nid] += 1
                    continue
                fout.write(f"{uid}\t{nid}\t1.0\t{ts}\n")
                ts += 1
            user_ts[uid] = ts

    # ---------- valid / test: labeled (후보 5개 + rating, 순서 셔플) ----------
    rng_v = random.Random(shuffle_seed)
    rng_t = random.Random(shuffle_seed + 1)

    def dump_eval_split(src_path: str, out_name: str, rng: random.Random) -> dict:
        """mind.test.inter 와 같이, 사용자당 5개 후보 행만 기록 (히스토리는 train에만 둠)."""
        gt: dict[str, str] = {}
        pos: dict[str, int] = {}
        skipped_rows = 0
        out_path = os.path.join(output_dir, out_name)
        row_no = 0
        with open(src_path, "r", encoding="utf-8") as fin, open(
            out_path, "w", encoding="utf-8"
        ) as fout:
            fout.write(INTER_HEADER)
            for line in fin:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    uid, _hist, cands, labels = parse_behavior_line(line)
                except ValueError:
                    continue
                if len(cands) < 5:
                    skipped_rows += 1
                    continue
                cands = cands[:5]
                all_users.add(uid)
                ratings = labels_from_candidates(cands, labels)
                ground = ground_truth_item(cands, ratings)
                pairs = [(c, ratings[i]) for i, c in enumerate(cands) if c in news_ids]
                if len(pairs) != len(cands):
                    skipped_rows += 1
                    continue
                row_no += 1
                rng.shuffle(pairs)
                base_ts = 2_000_000_000 + row_no
                gpos = next(i for i, (c, _) in enumerate(pairs) if c == ground)
                inst = f"{uid}__{row_no}"
                gt[inst] = ground
                pos[inst] = gpos
                for c, r in pairs:
                    fout.write(f"{uid}\t{c}\t{r}\t{base_ts}\n")
        return {"groundtruth": gt, "positions": pos, "skipped_rows": skipped_rows}

    meta_valid = dump_eval_split(valid_path, "mind_2000.valid.inter", rng_v)
    meta_test = dump_eval_split(test_path, "mind_2000.test.inter", rng_t)

    pkl_path = os.path.join(output_dir, "mind_2000_test_groundtruth.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(meta_test, f)

    # ---------- user ----------
    user_path = os.path.join(output_dir, "mind_2000.user")
    with open(user_path, "w", encoding="utf-8") as f:
        f.write(USER_HEADER)
        for uid in sorted(all_users, key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"{uid}\n")

    print(f"저장 완료: {output_dir}")
    print(f"  사용자 수: {len(all_users)}")
    print(f"  뉴스(아이템) 수: {len(news_info)}")
    if missing_news:
        top = sorted(missing_news.items(), key=lambda x: -x[1])[:10]
        print(f"  경고: 뉴스 메타에 없는 ID {len(missing_news)}종, 상위 스킵 횟수: {top}")
    print(f"  valid eval 스킵 행: {meta_valid.get('skipped_rows', 0)}")
    print(f"  test eval 스킵 행: {meta_test.get('skipped_rows', 0)}")
    print("실행 예:")
    print(
        "  python run_recbole.py --model=SASRec --dataset=mind_2000 "
        "--config_files=dataset/mind_2000/mind_2000.yaml"
    )


def main():
    ap = argparse.ArgumentParser(description="MIND_2000 TSV -> RecBole atomic")
    ap.add_argument(
        "--input_dir",
        default="MIND_2000",
        help="MIND_train.tsv 등이 있는 디렉터리",
    )
    ap.add_argument(
        "--output_dir",
        default="dataset/mind_2000",
        help="mind_2000.* 파일을 쓸 디렉터리",
    )
    ap.add_argument("--shuffle_seed", type=int, default=2023)
    args = ap.parse_args()
    convert(
        os.path.abspath(args.input_dir),
        os.path.abspath(args.output_dir),
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()

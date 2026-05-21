"""
Adressa_2000 폴더의 TSV를 RecBole atomic 형식으로 변환합니다 (mind_2000 과 동일 스키마).

입력 (기본 경로: ./Adressa_2000/)
  - Adressa_train.tsv:    user, clicked_news, candidate_news, clicked
  - Adressa_validation.tsv / Adressa_test.tsv: user, clicked_news, candidate_news (라벨 없으면 첫 후보=정답)
  - Adressa_news.tsv:     news_id, category, subcategory, title, body

출력: ./dataset/adressa_2000/
  adressa_2000.train.inter, adressa_2000.valid.inter, adressa_2000.test.inter
  adressa_2000.item, adressa_2000.user
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
from collections import defaultdict

import pandas as pd

DATASET_PREFIX = "adressa_2000"

INTER_HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"
ITEM_HEADER = "item_id:token\ttitle:token_seq\tcategory:token\tsubcategory:token\n"
USER_HEADER = "user_id:token\n"


def _clean_cell(s: str) -> str:
    return str(s).replace("\t", " ").replace("\n", " ").strip()


def load_news(news_path: str) -> tuple[dict[str, dict[str, str]], set[str]]:
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


def labels_from_candidates(cands: list[str], labels: list[int] | None) -> list[float]:
    if labels is not None:
        if len(labels) != len(cands):
            raise ValueError(f"후보 수({len(cands)})와 라벨 수({len(labels)}) 불일치")
        return [float(x) for x in labels]
    return [1.0] + [0.0] * (len(cands) - 1)


def ground_truth_item(cands: list[str], ratings: list[float]) -> str:
    for c, r in zip(cands, ratings):
        if r >= 0.5:
            return c
    return cands[0]


def convert(input_dir: str, output_dir: str, shuffle_seed: int = 2023) -> None:
    news_path = os.path.join(input_dir, "Adressa_news.tsv")
    train_path = os.path.join(input_dir, "Adressa_train.tsv")
    valid_path = os.path.join(input_dir, "Adressa_validation.tsv")
    test_path = os.path.join(input_dir, "Adressa_test.tsv")

    for p in (news_path, train_path, valid_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    os.makedirs(output_dir, exist_ok=True)
    news_info, news_ids = load_news(news_path)

    all_users: set[str] = set()
    missing_news: defaultdict[str, int] = defaultdict(int)

    item_path = os.path.join(output_dir, f"{DATASET_PREFIX}.item")
    with open(item_path, "w", encoding="utf-8") as f:
        f.write(ITEM_HEADER)
        for nid in sorted(news_info.keys(), key=lambda x: (len(x), x)):
            meta = news_info[nid]
            f.write(
                f"{nid}\t{meta['title']}\t{meta['category']}\t{meta['subcategory']}\n"
            )

    train_out = os.path.join(output_dir, f"{DATASET_PREFIX}.train.inter")
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

    rng_v = random.Random(shuffle_seed)
    rng_t = random.Random(shuffle_seed + 1)

    def dump_eval_split(src_path: str, out_name: str, rng: random.Random) -> dict:
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

    meta_valid = dump_eval_split(valid_path, f"{DATASET_PREFIX}.valid.inter", rng_v)
    meta_test = dump_eval_split(test_path, f"{DATASET_PREFIX}.test.inter", rng_t)

    pkl_path = os.path.join(output_dir, f"{DATASET_PREFIX}_test_groundtruth.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(meta_test, f)

    user_path = os.path.join(output_dir, f"{DATASET_PREFIX}.user")
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
        f"  python run_recbole.py --model=SASRec --dataset={DATASET_PREFIX} "
        f"--config_files=dataset/{DATASET_PREFIX}/{DATASET_PREFIX}.yaml"
    )


def main():
    ap = argparse.ArgumentParser(description="Adressa_2000 TSV -> RecBole atomic")
    ap.add_argument(
        "--input_dir",
        default="Adressa_2000",
        help="Adressa_train.tsv 등이 있는 디렉터리",
    )
    ap.add_argument(
        "--output_dir",
        default=f"dataset/{DATASET_PREFIX}",
        help=f"{DATASET_PREFIX}.* 파일을 쓸 디렉터리",
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

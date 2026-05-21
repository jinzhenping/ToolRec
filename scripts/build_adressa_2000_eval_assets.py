#!/usr/bin/env python3
"""
RecBole 학습 없이 adressa_2000 전용 평가 자산 생성:

1) dataset/prompts/test/adressa_2000_chat.pkl
2) dataset/prompts/test/adressa_2000_ui_token.pkl
3) dataset/prompts/test/adressa_2000_uid_alias.pkl
4) dataset/prompts/test/adressa_2000_eval_benchmark.tsv

의존성: pandas, numpy. torch / RecBole 불필요.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

DATASET_PREFIX = "adressa_2000"


def _strip_field_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.split(":")[0] for c in df.columns]
    return df


def _token_to_display_id(item_token: str) -> str:
    s = str(item_token)
    return s[1:] if s.startswith("N") and s[1:].isdigit() else s


def _format_title(raw) -> str:
    if isinstance(raw, (list, tuple)):
        return " ".join(str(x) for x in raw)
    return str(raw).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build adressa_2000 LLM-eval pickles + benchmark TSV."
    )
    ap.add_argument(
        "--dataset_dir",
        type=str,
        default=f"dataset/{DATASET_PREFIX}",
        help=f"{DATASET_PREFIX} atomic files directory",
    )
    ap.add_argument(
        "--pattern_json",
        type=str,
        default=f"dataset/prompts/test/{DATASET_PREFIX}_pattern.json",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="dataset/prompts/test",
    )
    ap.add_argument("--history_len", type=int, default=50)
    ap.add_argument("--max_instances", type=int, default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    inter_train = os.path.join(args.dataset_dir, f"{DATASET_PREFIX}.train.inter")
    inter_test = os.path.join(args.dataset_dir, f"{DATASET_PREFIX}.test.inter")
    item_path = os.path.join(args.dataset_dir, f"{DATASET_PREFIX}.item")

    for p in (inter_train, inter_test, item_path, args.pattern_json):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with open(args.pattern_json, "r", encoding="utf-8") as f:
        patterns = json.load(f)
    item_pat = patterns["item_pattern"]
    user_prefix = patterns["user_pattern"]

    item_df = _strip_field_names(pd.read_csv(item_path, sep="\t", dtype=str))
    item_profile: dict[str, str] = {}
    itemID_name: dict[str, str] = {}
    for _, row in item_df.iterrows():
        tok = str(row["item_id"]).strip()
        clean = _token_to_display_id(tok)
        title = _format_title(row.get("title", ""))
        cat = str(row.get("category", "")).strip()
        sub = str(row.get("subcategory", "")).strip()
        line = item_pat.format(iid=clean, title=title, category=cat, subcategory=sub)
        item_profile[clean] = line
        itemID_name[clean] = title

    train_df = _strip_field_names(pd.read_csv(inter_train, sep="\t"))
    test_df = _strip_field_names(pd.read_csv(inter_test, sep="\t"))

    for col in ("user_id", "item_id", "rating", "timestamp"):
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Expected column {col} in train/test .inter")

    train_df["rating"] = train_df["rating"].astype(float)
    train_df["timestamp"] = train_df["timestamp"].astype(float)
    test_df["rating"] = test_df["rating"].astype(float)
    test_df["timestamp"] = test_df["timestamp"].astype(float)

    train_pos = train_df[train_df["rating"] >= 0.5].sort_values(
        ["user_id", "timestamp", "item_id"]
    )

    def history_for_user(uid: str, ts_cutoff: float) -> list[str]:
        u = str(uid)
        g = train_pos[
            (train_pos["user_id"].astype(str) == u)
            & (train_pos["timestamp"] < ts_cutoff)
        ]
        items = [str(x) for x in g["item_id"].tolist()]
        return items[-args.history_len :] if items else []

    uid_iid: dict[str, str] = {}
    user_profile: dict[str, str] = {}
    uid_alias: dict[str, str] = {}
    tsv_lines: list[str] = []
    inst_counter: dict[str, int] = defaultdict(int)

    test_df = test_df.sort_values(["user_id", "timestamp"])
    written = 0
    for (uid, ts), g in test_df.groupby(["user_id", "timestamp"], sort=False):
        uid = str(uid)
        if len(g) != 5:
            continue
        pos = g[g["rating"] >= 0.5]
        if len(pos) != 1:
            continue
        neg = g[g["rating"] < 0.5]
        if len(neg) != 4:
            continue

        pos_item = str(pos.iloc[0]["item_id"])
        neg_items = sorted(str(x) for x in neg["item_id"].tolist())
        candidates = [pos_item] + neg_items

        k = inst_counter[uid]
        inst_counter[uid] += 1
        synth = f"{uid}__inst{k}"

        hist = history_for_user(uid, float(ts))
        tsv_lines.append(f"{synth}\t{' '.join(hist)}\t{' '.join(candidates)}\n")

        uid_iid[synth] = pos_item
        uid_alias[synth] = uid

        body = ""
        for htok in hist:
            clean = _token_to_display_id(htok)
            body += item_profile.get(clean, f"ID <{clean}>, (unknown item).\n")
        user_profile[synth] = user_prefix + body

        written += 1
        if args.max_instances is not None and written >= args.max_instances:
            break

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = DATASET_PREFIX

    chat_path = os.path.join(args.out_dir, f"{prefix}_chat.pkl")
    with open(chat_path, "wb") as f:
        pickle.dump((uid_iid, user_profile, item_profile, itemID_name), f)

    all_users = sorted(
        set(train_df["user_id"].astype(str)) | set(test_df["user_id"].astype(str)),
        key=lambda x: int(x),
    )
    all_items = sorted(
        item_df["item_id"].astype(str).unique().tolist(),
        key=lambda t: int(_token_to_display_id(t)),
    )

    pad = "<PAD>"
    user_token_id: dict[str, int] = {pad: 0}
    uid_list = [pad] + all_users
    for i, u in enumerate(uid_list):
        user_token_id[str(u)] = i
    user_id_token = np.array(uid_list, dtype=object)

    item_token_id: dict[str, int] = {pad: 0}
    iid_list = [pad] + all_items
    for i, it in enumerate(iid_list):
        item_token_id[str(it)] = i
    item_id_token = np.array(iid_list, dtype=object)

    token_out = os.path.join(args.out_dir, f"{prefix}_ui_token.pkl")
    with open(token_out, "wb") as f:
        pickle.dump((user_token_id, user_id_token, item_token_id, item_id_token), f)

    alias_path = os.path.join(args.out_dir, f"{prefix}_uid_alias.pkl")
    with open(alias_path, "wb") as f:
        pickle.dump(uid_alias, f)

    tsv_path = os.path.join(args.out_dir, f"{prefix}_eval_benchmark.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.writelines(tsv_lines)

    print(f"Wrote {chat_path} ({len(uid_iid)} instances)")
    print(f"Wrote {token_out}")
    print(f"Wrote {alias_path} ({len(uid_alias)} aliases)")
    print(f"Wrote {tsv_path} ({len(tsv_lines)} lines)")


if __name__ == "__main__":
    main()

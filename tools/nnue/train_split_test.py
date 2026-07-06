#!/usr/bin/env python3
"""
train_split_test.py — train.py の split_by_pid（pidグループ分割）の軽量検証。

torch は不要。numpy のみに依存する（train.py の import は sys.path 経由で行うが、
train.py 自体のトップレベルは torch を import しないため、torch 未インストール環境
でも実行できる）。

実行: python3 tools/nnue/train_split_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as train_mod  # noqa: E402


def make_records_with_pid(num_positions, syms_per_position=8):
    """pid付きの合成レコード列を作る（各pidにsyms_per_position個のレコード）。"""
    records = []
    for pid in range(num_positions):
        for sym in range(syms_per_position):
            records.append({"pid": pid, "sym": sym, "score": pid, "result": 0.5})
    return records


def make_records_without_pid(n):
    return [{"sym": 0, "score": i, "result": 0.5} for i in range(n)]


def test_pid_groups_never_split_across_train_val():
    records = make_records_with_pid(num_positions=50, syms_per_position=8)
    train_idx, val_idx, split_by = train_mod.split_by_pid(records, val_frac=0.1, seed=42)

    assert split_by == "pid"

    train_pids = {records[i]["pid"] for i in train_idx}
    val_pids = {records[i]["pid"] for i in val_idx}

    overlap = train_pids & val_pids
    assert not overlap, f"pid groups leaked across train/val: {overlap}"

    # every record must be assigned exactly once
    all_idx = set(train_idx.tolist()) | set(val_idx.tolist())
    assert all_idx == set(range(len(records)))
    assert len(train_idx) + len(val_idx) == len(records)

    # each pid's records must be entirely on one side
    for pid in train_pids | val_pids:
        rec_indices = [i for i, r in enumerate(records) if r["pid"] == pid]
        sides = set()
        for i in rec_indices:
            if i in set(train_idx.tolist()):
                sides.add("train")
            else:
                sides.add("val")
        assert len(sides) == 1, f"pid {pid} split across sides: {sides}"

    print("  [OK] test_pid_groups_never_split_across_train_val")


def test_val_frac_roughly_respected():
    num_positions = 200
    records = make_records_with_pid(num_positions=num_positions, syms_per_position=8)
    train_idx, val_idx, split_by = train_mod.split_by_pid(records, val_frac=0.1, seed=7)

    val_pids = {records[i]["pid"] for i in val_idx}
    train_pids = {records[i]["pid"] for i in train_idx}

    assert len(val_pids) + len(train_pids) == num_positions
    frac = len(val_pids) / num_positions
    # target 0.1, allow reasonable tolerance since rounding on small n
    assert abs(frac - 0.1) < 0.05, f"val fraction {frac} too far from 0.1"

    print(f"  [OK] test_val_frac_roughly_respected: val_frac={frac:.4f}")


def test_backward_compat_no_pid_falls_back_to_record_split():
    records = make_records_without_pid(100)
    train_idx, val_idx, split_by = train_mod.split_by_pid(records, val_frac=0.1, seed=1)

    assert split_by == "record"
    assert len(train_idx) + len(val_idx) == len(records)

    # matches split_train_val behavior exactly for the same seed/args
    expected_train, expected_val = train_mod.split_train_val(len(records), 0.1, 1)
    assert np.array_equal(np.sort(train_idx), np.sort(expected_train))
    assert np.array_equal(np.sort(val_idx), np.sort(expected_val))

    print("  [OK] test_backward_compat_no_pid_falls_back_to_record_split")


def test_deterministic_given_seed():
    records = make_records_with_pid(num_positions=30, syms_per_position=8)
    t1, v1, _ = train_mod.split_by_pid(records, val_frac=0.2, seed=99)
    t2, v2, _ = train_mod.split_by_pid(records, val_frac=0.2, seed=99)

    assert np.array_equal(t1, t2)
    assert np.array_equal(v1, v2)
    print("  [OK] test_deterministic_given_seed")


def test_single_sym_augment1_pid_equals_record_count():
    # augment=1 の場合、各pidは1レコードのみ。分割はpid単位だがレコード数=pid数なので
    # 実質的にレコード単位分割と同じ結果集合サイズになるはず。
    records = make_records_with_pid(num_positions=40, syms_per_position=1)
    train_idx, val_idx, split_by = train_mod.split_by_pid(records, val_frac=0.1, seed=5)

    assert split_by == "pid"
    assert len(train_idx) + len(val_idx) == 40
    print("  [OK] test_single_sym_augment1_pid_equals_record_count")


ALL_TESTS = [
    test_pid_groups_never_split_across_train_val,
    test_val_frac_roughly_respected,
    test_backward_compat_no_pid_falls_back_to_record_split,
    test_deterministic_given_seed,
    test_single_sym_augment1_pid_equals_record_count,
]


def run_all():
    failures = []
    for test_fn in ALL_TESTS:
        try:
            test_fn()
        except Exception as e:  # noqa: BLE001
            failures.append((test_fn.__name__, e))
            print(f"  [FAIL] {test_fn.__name__}: {e}")

    print()
    if failures:
        print(f"RESULT: {len(failures)} test(s) FAILED out of {len(ALL_TESTS)}")
        for name, e in failures:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print(f"RESULT: all {len(ALL_TESTS)} test(s) passed")


if __name__ == "__main__":
    run_all()

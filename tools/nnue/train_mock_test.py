#!/usr/bin/env python3
"""
train_mock_test.py — Step 3 (train.py) の mock 検証。

torch が無い環境でも走る numpy 専用テスト:
  1. reference_forward の手計算による健全性テスト
  2. カウント入力（重複 index）のテスト
  3. ClippedReLU 境界テスト
  4. bin/json ラウンドトリップ（書いて読んで forward）

torch がある環境でのみ実行される追加テスト:
  5. 合成データでの数エポック学習 + bin保存 + reference_forward との parity(1e-4以内)

pytest があれば `pytest tools/nnue/train_mock_test.py` で、無ければ
`python3 tools/nnue/train_mock_test.py` で直接実行できる。
"""

import json
import os
import struct
import sys
import tempfile

import numpy as np

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ─────────────────────────── reference forward (numpy only) ───────────────────────────

def clipped_relu(x):
    return np.clip(x, 0.0, 1.0)


def load_weights_from_bin(bin_path, meta):
    """Read gen*.bin per meta['tensors'] order, dtype float32 little-endian."""
    with open(bin_path, "rb") as f:
        data = f.read()

    weights = {}
    offset = 0
    for t in meta["tensors"]:
        name = t["name"]
        shape = t["shape"]
        count = 1
        for d in shape:
            count *= d
        nbytes = count * 4
        chunk = data[offset:offset + nbytes]
        if len(chunk) != nbytes:
            raise ValueError(f"Unexpected EOF reading tensor {name}: expected {nbytes} bytes, got {len(chunk)}")
        arr = np.frombuffer(chunk, dtype="<f4").reshape(shape).astype(np.float32)
        weights[name] = arr
        offset += nbytes

    if offset != len(data):
        raise ValueError(f"Trailing bytes in bin file: {len(data) - offset} bytes unconsumed")

    return weights


def build_dense_counts(indices, feature_dim):
    x = np.zeros((feature_dim,), dtype=np.float64)
    for idx in indices:
        if idx < 0 or idx >= feature_dim:
            raise ValueError(f"index {idx} out of range [0,{feature_dim})")
        x[idx] += 1.0
    return x


def reference_forward(weights, json_meta, indices, scalars):
    """
    Pure-numpy reference forward pass matching train.py's model exactly.
    Weights are read as float32 but all matmuls are done in float64
    (mirroring JS Number semantics for Step 4 parity).
    Returns (out, sigmoid_out) as Python floats.
    """
    feature_dim = json_meta["featureDim"]
    clip_min = json_meta["clipMin"]
    clip_max = json_meta["clipMax"]

    x = build_dense_counts(indices, feature_dim)  # float64
    s = np.asarray(scalars, dtype=np.float64)

    W1 = weights["W1"].astype(np.float64)
    b1 = weights["b1"].astype(np.float64)
    W2 = weights["W2"].astype(np.float64)
    b2 = weights["b2"].astype(np.float64)
    W3 = weights["W3"].astype(np.float64)
    b3 = weights["b3"].astype(np.float64)
    W4 = weights["W4"].astype(np.float64)
    b4 = weights["b4"].astype(np.float64)

    z1 = W1 @ x + b1
    a1 = np.clip(z1, clip_min, clip_max)

    h1 = np.concatenate([a1, s])
    z2 = W2 @ h1 + b2
    a2 = np.clip(z2, clip_min, clip_max)

    z3 = W3 @ a2 + b3
    a3 = np.clip(z3, clip_min, clip_max)

    z4 = W4 @ a3 + b4
    out = float(z4[0])
    sig = 1.0 / (1.0 + np.exp(-out))
    return out, float(sig)


# ─────────────────────────── test helpers ───────────────────────────

def make_toy_model(feature_dim=4, h1=3, scalar_dim=2, h2=2, h3=2):
    """Build small hand-chosen weights and a matching json meta dict."""
    rng = np.random.RandomState(0)
    W1 = (rng.rand(h1, feature_dim) - 0.5).astype(np.float32)
    b1 = (rng.rand(h1) - 0.5).astype(np.float32)
    W2 = (rng.rand(h2, h1 + scalar_dim) - 0.5).astype(np.float32)
    b2 = (rng.rand(h2) - 0.5).astype(np.float32)
    W3 = (rng.rand(h3, h2) - 0.5).astype(np.float32)
    b3 = (rng.rand(h3) - 0.5).astype(np.float32)
    W4 = (rng.rand(1, h3) - 0.5).astype(np.float32)
    b4 = (rng.rand(1) - 0.5).astype(np.float32)

    meta = {
        "format": "nnue-ukeja-v1",
        "generation": 999,
        "boardSize": 2,
        "featureDim": feature_dim,
        "scalarDim": scalar_dim,
        "maxDepth": 8,
        "hidden": [h1, h2, h3],
        "concatScalarsAfterLayer": 1,
        "activation": "clipped_relu",
        "clipMin": 0.0,
        "clipMax": 1.0,
        "scoreScale": 100.0,
        "lambda": 0.6,
        "dtype": "float32",
        "byteOrder": "little-endian",
        "tensors": [
            {"name": "W1", "shape": [h1, feature_dim]},
            {"name": "b1", "shape": [h1]},
            {"name": "W2", "shape": [h2, h1 + scalar_dim]},
            {"name": "b2", "shape": [b2.shape[0]]},
            {"name": "W3", "shape": [h3, h2]},
            {"name": "b3", "shape": [h3]},
            {"name": "W4", "shape": [1, h3]},
            {"name": "b4", "shape": [1]},
        ],
    }

    weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
    return weights, meta


def write_bin_from_weights(path, meta, weights):
    order = [t["name"] for t in meta["tensors"]]
    with open(path, "wb") as f:
        for name in order:
            arr = weights[name].astype("<f4")
            f.write(np.ascontiguousarray(arr).tobytes())


# ─────────────────────────── tests: torch-independent ───────────────────────────

def test_hand_computed_forward():
    """Tiny hand-chosen weights; compute forward manually and compare to reference_forward."""
    # feature_dim=2, h1=1, scalar_dim=1, h2=1, h3=1
    feature_dim, h1, scalar_dim, h2, h3 = 2, 1, 1, 1, 1

    W1 = np.array([[2.0, -1.0]], dtype=np.float32)  # shape (1,2)
    b1 = np.array([0.5], dtype=np.float32)
    W2 = np.array([[1.0, 3.0]], dtype=np.float32)   # shape (1, h1+scalar_dim=2)
    b2 = np.array([-0.2], dtype=np.float32)
    W3 = np.array([[0.5]], dtype=np.float32)        # shape (1,1)
    b3 = np.array([0.1], dtype=np.float32)
    W4 = np.array([[2.0]], dtype=np.float32)        # shape (1,1)
    b4 = np.array([-0.5], dtype=np.float32)

    meta = {
        "featureDim": feature_dim, "clipMin": 0.0, "clipMax": 1.0,
        "tensors": [
            {"name": "W1", "shape": [h1, feature_dim]}, {"name": "b1", "shape": [h1]},
            {"name": "W2", "shape": [h2, h1 + scalar_dim]}, {"name": "b2", "shape": [h2]},
            {"name": "W3", "shape": [h3, h2]}, {"name": "b3", "shape": [h3]},
            {"name": "W4", "shape": [1, h3]}, {"name": "b4", "shape": [1]},
        ],
    }
    weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}

    indices = [0, 1]  # x = [1, 1]
    scalars = [0.25]

    # Manual computation:
    # x = [1, 1]
    # z1 = 2*1 + (-1)*1 + 0.5 = 1.5  -> clip[0,1] -> 1.0
    x = np.array([1.0, 1.0])
    z1 = W1.astype(np.float64) @ x + b1.astype(np.float64)
    assert abs(z1[0] - 1.5) < 1e-6
    a1 = np.clip(z1, 0.0, 1.0)
    assert abs(a1[0] - 1.0) < 1e-6

    # h1 = concat(a1, s) = [1.0, 0.25]
    # z2 = 1*1.0 + 3*0.25 - 0.2 = 1.55 -> clip -> 1.0
    h1_cat = np.concatenate([a1, np.array(scalars)])
    z2 = W2.astype(np.float64) @ h1_cat + b2.astype(np.float64)
    assert abs(z2[0] - 1.55) < 1e-6
    a2 = np.clip(z2, 0.0, 1.0)
    assert abs(a2[0] - 1.0) < 1e-6

    # z3 = 0.5*1.0 + 0.1 = 0.6 -> clip -> 0.6
    z3 = W3.astype(np.float64) @ a2 + b3.astype(np.float64)
    assert abs(z3[0] - 0.6) < 1e-6
    a3 = np.clip(z3, 0.0, 1.0)
    assert abs(a3[0] - 0.6) < 1e-6

    # out = 2*0.6 - 0.5 = 0.7
    z4 = W4.astype(np.float64) @ a3 + b4.astype(np.float64)
    expected_out = 0.7
    assert abs(z4[0] - expected_out) < 1e-6

    out, sig = reference_forward(weights, meta, indices, scalars)
    assert abs(out - expected_out) < 1e-6, f"expected {expected_out}, got {out}"
    expected_sig = 1.0 / (1.0 + np.exp(-expected_out))
    assert abs(sig - expected_sig) < 1e-6

    print(f"  [OK] test_hand_computed_forward: out={out:.6f} sigmoid={sig:.6f}")


def test_duplicate_indices_count():
    """indices=[5,5,5] should produce x[5] == 3, not one-hot."""
    feature_dim = 10
    x = build_dense_counts([5, 5, 5, 2], feature_dim)
    assert x[5] == 3.0, f"expected x[5]==3, got {x[5]}"
    assert x[2] == 1.0
    assert x.sum() == 4.0
    print("  [OK] test_duplicate_indices_count")


def test_clipped_relu_boundaries():
    vals = np.array([-5.0, -0.001, 0.0, 0.5, 1.0, 1.001, 10.0])
    out = clipped_relu(vals)
    expected = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    assert np.allclose(out, expected), f"got {out}"
    print("  [OK] test_clipped_relu_boundaries")


def test_bin_roundtrip_matches_reference():
    """Write toy weights to bin+json, read them back, verify reference_forward matches
    a directly-computed value using the loaded (not original) weight arrays."""
    weights, meta = make_toy_model(feature_dim=6, h1=4, scalar_dim=2, h2=3, h3=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "toy.bin")
        json_path = os.path.join(tmpdir, "toy.json")
        write_bin_from_weights(bin_path, meta, weights)
        with open(json_path, "w") as f:
            json.dump(meta, f)

        with open(json_path) as f:
            loaded_meta = json.load(f)
        loaded_weights = load_weights_from_bin(bin_path, loaded_meta)

        for name in weights:
            assert np.allclose(weights[name], loaded_weights[name], atol=1e-7), f"mismatch in {name}"

        indices = [0, 1, 1, 3]
        scalars = [0.3, 0.7]
        out1, sig1 = reference_forward(weights, meta, indices, scalars)
        out2, sig2 = reference_forward(loaded_weights, loaded_meta, indices, scalars)
        assert abs(out1 - out2) < 1e-6, f"{out1} vs {out2}"
        assert abs(sig1 - sig2) < 1e-6

        # empty indices edge case
        out3, sig3 = reference_forward(loaded_weights, loaded_meta, [], scalars)
        assert np.isfinite(out3) and np.isfinite(sig3)

    print(f"  [OK] test_bin_roundtrip_matches_reference: out={out1:.6f}")


# ─────────────────────────── tests: torch-dependent (parity) ───────────────────────────

def test_torch_training_and_parity():
    if not HAS_TORCH:
        print("  [SKIP] test_torch_training_and_parity: torch not installed")
        return

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train as train_mod

    rng = np.random.RandomState(42)
    feature_dim = 16  # e.g. boardSize=1 is invalid; just use a synthetic dim for smoke test
    scalar_dim = 4
    n = 300

    synthetic_records = []
    for _ in range(n):
        num_active = rng.randint(0, 5)
        indices = rng.randint(0, feature_dim, size=num_active).tolist()
        scalars = rng.rand(scalar_dim).tolist()
        score = float(rng.randint(-130, 131))
        result = float(rng.choice([0, 0.5, 1]))
        synthetic_records.append({
            "size": 2, "pov": "white", "ply": 0, "sym": 0,
            "indices": indices, "scalars": scalars, "score": score, "result": result,
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "synthetic.jsonl")
        with open(data_path, "w") as f:
            for rec in synthetic_records:
                f.write(json.dumps(rec) + "\n")

        args = train_mod.parse_args.__wrapped__ if hasattr(train_mod.parse_args, "__wrapped__") else None

        class Args:
            pass

        a = Args()
        a.data = data_path
        a.out_dir = tmpdir
        a.generation = 999
        a.score_scale = 100.0
        a.lam = 0.6
        a.hidden1 = 8
        a.hidden2 = 6
        a.hidden3 = 6
        a.epochs = 3
        a.patience = 10
        a.batch_size = 32
        a.lr = 1e-3
        a.val_frac = 0.1
        a.seed = 1
        a.device = "cpu"
        a.golden_samples = 5

        train_mod.set_seed(a.seed)
        records = train_mod.load_records(a.data)
        board_size = records[0]["size"]
        max_depth = 8
        actual_feature_dim = board_size * board_size * max_depth * 2  # matches train.py convention

        # Rebuild synthetic indices within actual_feature_dim bounds
        for r in records:
            r["indices"] = [i % actual_feature_dim for i in r["indices"]]

        x_dense, scalars_arr, y = train_mod.build_dataset(
            records, actual_feature_dim, scalar_dim, a.score_scale, a.lam
        )
        train_idx, val_idx = train_mod.split_train_val(len(records), a.val_frac, a.seed)
        device = torch.device("cpu")
        model = train_mod.build_model(actual_feature_dim, scalar_dim, a.hidden1, a.hidden2, a.hidden3)

        model, best_val_loss, last_train_loss, best_epoch, epochs_run = train_mod.train_model(
            model, device, x_dense, scalars_arr, y, train_idx, val_idx, a
        )
        assert np.isfinite(best_val_loss)
        assert np.isfinite(last_train_loss)
        print(f"  [OK] synthetic training ran: trainLoss={last_train_loss:.6f} valLoss={best_val_loss:.6f} epochsRun={epochs_run}")

        bin_path = os.path.join(tmpdir, "gen999.bin")
        json_path = os.path.join(tmpdir, "gen999.json")
        train_mod.write_bin(bin_path, model)

        meta = {
            "featureDim": actual_feature_dim,
            "clipMin": 0.0,
            "clipMax": 1.0,
            "tensors": [
                {"name": "W1", "shape": [a.hidden1, actual_feature_dim]},
                {"name": "b1", "shape": [a.hidden1]},
                {"name": "W2", "shape": [a.hidden2, a.hidden1 + scalar_dim]},
                {"name": "b2", "shape": [a.hidden2]},
                {"name": "W3", "shape": [a.hidden3, a.hidden2]},
                {"name": "b3", "shape": [a.hidden3]},
                {"name": "W4", "shape": [1, a.hidden3]},
                {"name": "b4", "shape": [1]},
            ],
        }
        train_mod.write_json(json_path, meta)

        loaded_meta = json.load(open(json_path))
        loaded_weights = load_weights_from_bin(bin_path, loaded_meta)

        model.eval()
        max_abs_diff = 0.0
        with torch.no_grad():
            for rec in records[:30]:
                x = train_mod.build_dense_counts(rec["indices"], actual_feature_dim)
                x_t = torch.tensor([x], dtype=torch.float32)
                s_t = torch.tensor([rec["scalars"]], dtype=torch.float32)
                torch_out = float(model(x_t, s_t).item())

                ref_out, _ = reference_forward(loaded_weights, loaded_meta, rec["indices"], rec["scalars"])
                diff = abs(torch_out - ref_out)
                max_abs_diff = max(max_abs_diff, diff)

        assert max_abs_diff < 1e-4, f"parity failed: max_abs_diff={max_abs_diff}"
        print(f"  [OK] test_torch_training_and_parity: max_abs_diff={max_abs_diff:.8f} (< 1e-4)")


# ─────────────────────────── runner ───────────────────────────

ALL_TESTS = [
    test_hand_computed_forward,
    test_duplicate_indices_count,
    test_clipped_relu_boundaries,
    test_bin_roundtrip_matches_reference,
    test_torch_training_and_parity,
]


def run_all():
    print(f"torch available: {HAS_TORCH}")
    if not HAS_TORCH:
        print("NOTE: torch is not installed in this environment. Torch-dependent parity test will be SKIPPED.")
        print("      Full training (real gen000.jsonl) must be run by the parent agent with torch installed.")
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
        print(f"RESULT: all {len(ALL_TESTS)} test(s) passed (or skipped where torch unavailable)")


if __name__ == "__main__":
    run_all()

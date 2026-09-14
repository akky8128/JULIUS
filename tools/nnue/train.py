#!/usr/bin/env python3
"""
train.py — NNUE 評価関数の学習スクリプト（Step 3）

Step 1 (js/ai/nnue/features.js) と Step 2 (tools/nnue/genData.mjs) が生成した
教師データ JSONL を読み込み、PyTorch で NNUE 型のネットワークを学習する。

出力される gen{N}.bin / gen{N}.json / gen{N}.golden.json は、Step 4 (JS 側での
forward 再現・検証) の入力契約となるため、数式・バイナリ配置・JSON スキーマは
このファイルとの厳密な整合を保つこと。

モデル形状:
    x(featureDim, カウントベクトル) --Linear(W1,b1)--> ClippedReLU --> a1(H1)
    concat(a1, scalars(scalarDim)) --Linear(W2,b2)--> ClippedReLU --> a2(H2)
    a2 --Linear(W3,b3)--> ClippedReLU --> a3(H3)
    a3 --Linear(W4,b4)--> out (線形, スカラー)

教師信号:
    y = lambda * sigmoid(score / scoreScale) + (1 - lambda) * result
    loss = MSE( sigmoid(out), y )
"""

import argparse
import json
import os
import random
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="Train NNUE evaluation network for Ukeja Layer.")
    p.add_argument("--data", type=str, required=True, help="Path to training JSONL (e.g. tools/nnue/data/gen000.jsonl)")
    p.add_argument("--out-dir", type=str, default="models", help="Output directory for gen{N}.bin/json/golden.json")
    p.add_argument("--generation", type=int, default=1, help="Generation number N (output files gen00N.*)")
    p.add_argument("--score-scale", type=float, default=100.0, help="S in y = lambda*sigmoid(score/S) + (1-lambda)*result")
    p.add_argument("--score-clip", type=float, default=2000.0, help="Clamp |score| to this value before sigmoid (removes ±mate outliers; must exceed normal heuristic range)")
    p.add_argument("--lambda", dest="lam", type=float, default=0.6, help="lambda in the teacher signal blend")
    # 既定は 96/24/24。64/16/16 は初期の小規模データでは最速かつ最強だったが、
    # データが約120万レコードに増えた段階で 64/16/16 は容量飽和して頭打ち（gen015）。
    # 96/24/24 に拡大した gen016 が gen014[64/16/16] に直接対決で勝ち越し（64.8%）、
    # プラトーを突破した（128/32/32 は過大で互角＝無駄）。eval は 64/16/16 の約3倍遅い
    # トレードオフがあるため、最速枠が必要なときは 64/16/16 を明示指定する。
    p.add_argument("--hidden1", type=int, default=96, help="H1: size of first hidden layer")
    p.add_argument("--hidden2", type=int, default=24, help="H2: size of second hidden layer")
    p.add_argument("--hidden3", type=int, default=24, help="H3: size of third hidden layer")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--golden-samples", type=int, default=24)
    p.add_argument("--stack-encoder-dim", type=int, default=0,
                   help="E: shared per-cell stack-encoder output dim (0 = disabled, current behavior)")
    p.add_argument("--scalar-in-l1", action="store_true",
                   help="scalars を L1(a1) 経路にも注入する(既存のL2直結concatは維持、追加の経路)")
    p.add_argument("--no-fc-path", action="store_true",
                   help="従来の全結合L1(a1)経路を削除し、スタックエンコーダのpooled出力のみで"
                        "h1を構成する(純エンコーダ版)。--stack-encoder-dim>0が必須。")
    p.add_argument("--policy-head", action="store_true",
                   help="AlphaZero P1: 2ヘッド化。h1 concat の scalars の後ろにターン文脈(26)を追加し、"
                        "L3 出力から policy logits(81) を出すヘッド Wp/bp をテンソル列の末尾に追記する。"
                        "無指定なら従来と完全同一（後方互換）。")
    p.add_argument("--turn-context-dim", type=int, default=26,
                   help="ターン文脈ベクトルの次元（--policy-head 時のみ使用）")
    p.add_argument("--num-actions", type=int, default=81,
                   help="policy head の出力アクション数（--policy-head 時のみ使用）")
    p.add_argument("--gen-tag", type=str, default=None,
                   help="出力ファイル名タグの上書き（例: genAZ000）。省略時は gen{generation:03d}")
    p.add_argument("--init-bin", type=str, default=None,
                   help="warm-start: 既存モデルの .bin から重みを読み込んでから学習を継続する"
                        "（自己対戦ループの継続学習用。アーキ引数は読み込むモデルと一致必須）。")
    p.add_argument("--encoder-max-pool", action="store_true",
                   help="per-cellエンコーダのpoolingをsum-poolのみからsum-pool+max-poolの"
                        "concatに拡張する(encoder出力E次元がh1では2E次元になる)。"
                        "--stack-encoder-dim>0が必須。Wenc/bencは共有のまま追加パラメータなし。")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def resolve_device(requested):
    import torch
    if requested != "auto":
        if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print(f"[train] Warning: --device mps requested but not available. Falling back to cpu.")
            return torch.device("cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            print(f"[train] Warning: --device cuda requested but not available. Falling back to cpu.")
            return torch.device("cpu")
        return torch.device(requested)
    # auto: mps -> cuda -> cpu
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[train] device=auto -> mps")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[train] device=auto -> cuda")
        return torch.device("cuda")
    print("[train] device=auto -> cpu")
    return torch.device("cpu")


def load_records(path):
    """Read JSONL training records. Returns a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            for key in ("size", "indices", "scalars"):
                if key not in rec:
                    raise ValueError(f"Missing key '{key}' at {path}:{line_no}")
            # 価値目標は「score+result（P2/NNUE）」または「z（P4 自己対戦, mover視点±1/0.5）」。
            has_sr = ("score" in rec and "result" in rec)
            has_z = ("z" in rec)
            if not (has_sr or has_z):
                raise ValueError(f"Record needs 'score'+'result' or 'z' at {path}:{line_no}")
            records.append(rec)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def build_dense_counts(indices, feature_dim):
    """Build a dense count vector: x[i] += 1 for each occurrence in indices (not one-hot)."""
    x = [0.0] * feature_dim
    for idx in indices:
        if idx < 0 or idx >= feature_dim:
            raise ValueError(f"Feature index {idx} out of range [0,{feature_dim})")
        x[idx] += 1.0
    return x


def build_dataset(records, feature_dim, scalar_dim, score_scale, lam, score_clip=2000.0):
    import numpy as np
    n = len(records)
    x_dense = np.zeros((n, feature_dim), dtype=np.float32)
    scalars = np.zeros((n, scalar_dim), dtype=np.float32)
    scores = np.zeros((n,), dtype=np.float64)
    results = np.zeros((n,), dtype=np.float64)
    # z 直接目標（P4 自己対戦）: sigmoid(out)=mover視点勝率なので y=(z+1)/2。
    y_from_z = np.full((n,), np.nan, dtype=np.float64)

    for i, rec in enumerate(records):
        for idx in rec["indices"]:
            if idx < 0 or idx >= feature_dim:
                raise ValueError(f"Feature index {idx} out of range [0,{feature_dim}) at record {i}")
            x_dense[i, idx] += 1.0
        sc = rec["scalars"]
        if len(sc) != scalar_dim:
            raise ValueError(f"Record {i}: scalars length {len(sc)} != expected {scalar_dim}")
        scalars[i, :] = sc
        # Clamp scores to a safe range before sigmoid. Terminal mate scores
        # (±1e9) otherwise dwarf the normal heuristic range and are pure
        # outliers; clamp keeps them decisively winning/losing without letting
        # their raw magnitude leak into any score-scale retuning.
        if "z" in rec and not ("score" in rec and "result" in rec):
            y_from_z[i] = (float(rec["z"]) + 1.0) / 2.0
        else:
            raw_score = float(rec["score"])
            raw_score = max(-score_clip, min(score_clip, raw_score))
            scores[i] = raw_score
            results[i] = float(rec["result"])

    # y = lambda * sigmoid(score/S) + (1-lambda) * result（score/result 系）
    sig = 1.0 / (1.0 + np.exp(-scores / score_scale))
    y = lam * sig + (1.0 - lam) * results
    # z 系レコードは y_from_z を優先（P4 自己対戦）
    zmask = ~np.isnan(y_from_z)
    y[zmask] = y_from_z[zmask]
    y = y.astype(np.float32)

    return x_dense, scalars, y


class ClippedReLU:
    """Marker class kept only for documentation; actual op done inline via torch.clamp."""
    pass


def build_model(feature_dim, scalar_dim, h1, h2, h3, stack_encoder_dim=0, scalar_in_l1=False, no_fc_path=False,
                 encoder_max_pool=False, policy_head=False, turn_context_dim=26, num_actions=81):
    """
    stack_encoder_dim > 0 の場合、共有重みの per-cell スタックエンコーダ
    (Linear(16 -> E) をセル毎に同じ重みで適用し、16セル分を sum-pool する)
    を追加する。encoder 出力 p(E) は a1 と scalars の間に concat される:
        h1 = concat(a1[H1], p[E], scalars[scalarDim])
    stack_encoder_dim=0 (デフォルト) の場合は従来通り encoder なし。

    scalar_in_l1=True の場合、scalars を a1(L1)経路にも追加注入する
    (bias なし線形 ls1: scalarDim -> H1 を l1 の出力に加算してから clippedReLU)。
    既存の L2 直結 concat は維持したまま追加する経路であり、スカラーが深い
    非線形(a2/a3)にも早期に影響できるようにする狙い。
    """
    import torch
    import torch.nn as nn

    CELL_SLOT_DIM = 16  # MAX_DEPTH(8) * 2 colors
    num_cells = feature_dim // CELL_SLOT_DIM

    class NNUEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.stack_encoder_dim = stack_encoder_dim
            self.scalar_in_l1 = scalar_in_l1
            self.no_fc_path = no_fc_path
            self.num_cells = num_cells
            self.encoder_max_pool = encoder_max_pool
            if not no_fc_path:
                self.l1 = nn.Linear(feature_dim, h1)
            if stack_encoder_dim > 0:
                self.enc = nn.Linear(CELL_SLOT_DIM, stack_encoder_dim)
            if scalar_in_l1 and not no_fc_path:
                self.ls1 = nn.Linear(scalar_dim, h1, bias=False)
            enc_contrib = stack_encoder_dim * (2 if (stack_encoder_dim > 0 and encoder_max_pool) else 1)
            self.policy_head = policy_head
            self.turn_context_dim = turn_context_dim if policy_head else 0
            l2_in = (0 if no_fc_path else h1) + enc_contrib + scalar_dim + self.turn_context_dim
            self.l2 = nn.Linear(l2_in, h2)
            self.l3 = nn.Linear(h2, h3)
            self.l4 = nn.Linear(h3, 1)
            if policy_head:
                # policy head: L3(h3) -> num_actions logits（JS側 Wp/bp に対応）
                self.lp = nn.Linear(h3, num_actions)

        def forward(self, x, s, tc=None):
            parts = []
            if not self.no_fc_path:
                pre1 = self.l1(x)
                if self.scalar_in_l1:
                    pre1 = pre1 + self.ls1(s)
                a1 = torch.clamp(pre1, 0.0, 1.0)
                parts.append(a1)
            if self.stack_encoder_dim > 0:
                # x[:, :feature_dim] -> [B, num_cells, 16] (共有重み Wenc/benc を全セルに適用)
                per_cell = x.view(x.shape[0], self.num_cells, CELL_SLOT_DIM)
                e_cell = torch.clamp(self.enc(per_cell), 0.0, 1.0)  # [B, num_cells, E]
                pooled_sum = e_cell.sum(dim=1)  # sum-pool over cells -> [B, E]
                parts.append(pooled_sum)
                if self.encoder_max_pool:
                    pooled_max = e_cell.max(dim=1).values  # max-pool over cells -> [B, E]
                    parts.append(pooled_max)
            parts.append(s)
            if self.policy_head:
                # ターン文脈は scalars の後ろに concat（JS側 network.js と同一位置）
                if tc is None:
                    tc = torch.zeros(x.shape[0], self.turn_context_dim,
                                     dtype=x.dtype, device=x.device)
                parts.append(tc)
            h1_cat = torch.cat(parts, dim=1)
            a2 = torch.clamp(self.l2(h1_cat), 0.0, 1.0)
            a3 = torch.clamp(self.l3(a2), 0.0, 1.0)
            out = self.l4(a3)
            if self.policy_head:
                policy_logits = self.lp(a3)
                return out.squeeze(-1), policy_logits
            return out.squeeze(-1)

    return NNUEModel()


def split_train_val(n, val_frac, seed):
    import numpy as np
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def split_by_pid(records, val_frac, seed):
    """
    Group-aware split: records sharing the same 'pid' (i.e. the D4-augmented
    sym=0..7 copies of one original position) are always kept together in
    either train or val. This avoids leakage where near-duplicate symmetric
    copies of the same position end up split across train/val.

    Falls back to plain record-level split (with a warning) if records do
    not carry a 'pid' key (older datasets generated before pid support).

    @param records: list of dicts (each optionally containing 'pid')
    @param val_frac: float, target fraction of *positions* (pids) in val
    @param seed: int, RNG seed for the pid shuffle
    @returns: (train_idx, val_idx) as numpy int arrays indexing into `records`
              and a `split_by` string: "pid" or "record"
    """
    import numpy as np

    n = len(records)
    has_pid = all("pid" in r for r in records)

    if not has_pid:
        print("[train] Warning: records do not contain 'pid' (older dataset format). "
              "Falling back to record-level train/val split (may leak D4-augmented "
              "copies of the same position across train/val).")
        train_idx, val_idx = split_train_val(n, val_frac, seed)
        return train_idx, val_idx, "record"

    pids = np.array([r["pid"] for r in records])
    unique_pids = np.unique(pids)
    n_pids = len(unique_pids)

    rng = np.random.RandomState(seed)
    shuffled_pids = unique_pids.copy()
    rng.shuffle(shuffled_pids)

    n_val_pids = max(1, int(round(n_pids * val_frac))) if n_pids > 1 else 0
    val_pid_set = set(shuffled_pids[:n_val_pids].tolist())

    val_mask = np.array([pid in val_pid_set for pid in pids])
    train_mask = ~val_mask

    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]

    return train_idx, val_idx, "pid"


def soft_cross_entropy(logits, target_dist, eps=1e-12):
    """ソフト目標 policy CE: -Σ_a π_a · log_softmax(logits)_a を各行で計算し平均する。
    π が one-hot のとき通常の hard CrossEntropyLoss(index=argmax) と一致する。
    target_dist の行和が 0 の行（教師なし）は損失に寄与させない。
    @param logits: [B, A] 生ロジット
    @param target_dist: [B, A] 非負・行和1（教師なし行は行和0でも可）
    @returns スカラー（教師あり行の平均）
    """
    import torch
    import torch.nn.functional as F
    logp = F.log_softmax(logits, dim=1)
    per_row = -(target_dist * logp).sum(dim=1)  # [B]
    row_mass = target_dist.sum(dim=1)  # [B]
    mask = row_mass > eps
    if not bool(mask.any()):
        return logits.sum() * 0.0  # 教師なし: 勾配ゼロのスカラー
    return per_row[mask].mean()


def train_model(model, device, x_dense, scalars, y, train_idx, val_idx, args,
                policy_targets=None, turn_contexts=None, policy_dist=None):
    """policy_targets: int64 array with action index or -1（ハード目標, P2蒸留）。
    policy_dist: float32 [n, numActions]（ソフト目標分布, P4 自己対戦）。存在すれば
      soft_cross_entropy を使い、無ければ policy_targets のハード CE にフォールバックする
      （後方互換・フィールド存在でゲート）。
    turn_contexts: float32 [n, turnContextDim]（無いレコードはゼロ行）。
    いずれも policy head モデルのみ使用。"""
    import torch
    import torch.nn as nn
    import numpy as np

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    has_policy = bool(getattr(model, "policy_head", False))
    use_soft = has_policy and policy_dist is not None
    ce_fn = nn.CrossEntropyLoss() if (has_policy and not use_soft) else None
    pt_train = (torch.from_numpy(policy_targets[train_idx])
                if (has_policy and not use_soft and policy_targets is not None) else None)
    pt_val = (torch.from_numpy(policy_targets[val_idx]).to(device)
              if (has_policy and not use_soft and policy_targets is not None) else None)
    pd_train = (torch.from_numpy(policy_dist[train_idx]) if use_soft else None)
    pd_val = (torch.from_numpy(policy_dist[val_idx]).to(device) if use_soft else None)
    tc_train = (torch.from_numpy(turn_contexts[train_idx])
                if (has_policy and turn_contexts is not None) else None)
    tc_val = (torch.from_numpy(turn_contexts[val_idx]).to(device)
              if (has_policy and turn_contexts is not None) else None)

    x_train = torch.from_numpy(x_dense[train_idx])
    s_train = torch.from_numpy(scalars[train_idx])
    y_train = torch.from_numpy(y[train_idx])

    x_val = torch.from_numpy(x_dense[val_idx]).to(device)
    s_val = torch.from_numpy(scalars[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)

    n_train = x_train.shape[0]
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0
    last_train_loss = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss_sum = 0.0
        epoch_count = 0
        for start in range(0, n_train, args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            xb = x_train[batch_idx].to(device)
            sb = s_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)

            optimizer.zero_grad()
            if has_policy:
                tcb = tc_train[batch_idx].to(device) if tc_train is not None else None
                out, pol = model(xb, sb, tcb)
                pred = torch.sigmoid(out)
                loss = loss_fn(pred, yb)
                if use_soft:
                    pdb = pd_train[batch_idx].to(device)
                    loss = loss + soft_cross_entropy(pol, pdb)
                elif pt_train is not None:
                    tb = pt_train[batch_idx].to(device)
                    mask = tb >= 0
                    if mask.any():
                        loss = loss + ce_fn(pol[mask], tb[mask])
            else:
                out = model(xb, sb)
                pred = torch.sigmoid(out)
                loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * xb.shape[0]
            epoch_count += xb.shape[0]

        train_loss = epoch_loss_sum / max(1, epoch_count)
        last_train_loss = train_loss

        model.eval()
        val_top1 = None
        with torch.no_grad():
            val_ce = None
            if has_policy:
                val_out, val_pol = model(x_val, s_val, tc_val)
                if use_soft:
                    vmask = pd_val.sum(dim=1) > 1e-12
                    if bool(vmask.any()):
                        # ソフト目標: top-1 は π の argmax をターゲット近似として使う
                        pred_actions = val_pol[vmask].argmax(dim=1)
                        tgt_actions = pd_val[vmask].argmax(dim=1)
                        val_top1 = float((pred_actions == tgt_actions).float().mean().item())
                        val_ce = float(soft_cross_entropy(val_pol[vmask], pd_val[vmask]).item())
                elif pt_val is not None:
                    vmask = pt_val >= 0
                    if vmask.any():
                        # policy top-1 accuracy（合法手マスクなしの生 argmax）
                        pred_actions = val_pol[vmask].argmax(dim=1)
                        val_top1 = float((pred_actions == pt_val[vmask]).float().mean().item())
                        val_ce = float(ce_fn(val_pol[vmask], pt_val[vmask]).item())
            else:
                val_out = model(x_val, s_val)
            val_pred = torch.sigmoid(val_out)
            val_loss = loss_fn(val_pred, y_val).item()
            # policy 教師がある場合、モデル選択/早期停止は value+policy の合成 val loss で行う
            # （value のみだと policy 収束前に停止し得るため）
            if val_ce is not None:
                val_loss = val_loss + val_ce

        top1_str = (f"  val_policy_top1={val_top1:.4f}  val_ce={val_ce:.4f}"
                    if val_top1 is not None else "")
        print(f"[train] epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{top1_str}")

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model._best_val_top1 = val_top1  # ベストepoch時点の policy top-1（レポート用）
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"[train] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_val_loss = val_loss
        best_epoch = epoch

    return model, best_val_loss, last_train_loss, best_epoch, epoch


def tensor_order(stack_encoder_dim=0, scalar_in_l1=False, no_fc_path=False, policy_head=False):
    """write_bin / load_bin で共有する固定テンソル順序。この2つが必ず同一順序を使うよう
    一元化する（順序がズレると .bin の読み書きが静かに壊れるため）。"""
    if no_fc_path:
        order = ["l2.weight", "l2.bias", "l3.weight", "l3.bias", "l4.weight", "l4.bias"]
    else:
        order = ["l1.weight", "l1.bias", "l2.weight", "l2.bias",
                  "l3.weight", "l3.bias", "l4.weight", "l4.bias"]
    if stack_encoder_dim > 0:
        order += ["enc.weight", "enc.bias"]
    if scalar_in_l1 and not no_fc_path:
        order += ["ls1.weight"]
    if policy_head:
        # policy head (Wp/bp) は常にテンソル列の最後（既存順序を不変に保つ）
        order += ["lp.weight", "lp.bias"]
    return order


def load_bin_into(model, bin_path, stack_encoder_dim=0, scalar_in_l1=False, no_fc_path=False, policy_head=False):
    """既存 .bin（write_bin と同一順序の float32-LE）をモデルへ読み込む（warm-start 用）。
    モデルの state_dict の shape で各テンソルを切り出す。サイズ不一致は即エラー。"""
    import numpy as np
    import torch
    sd = model.state_dict()
    order = tensor_order(stack_encoder_dim, scalar_in_l1, no_fc_path, policy_head)
    data = np.fromfile(bin_path, dtype="<f4")
    offset = 0
    for key in order:
        shape = tuple(sd[key].shape)
        n = int(np.prod(shape)) if len(shape) > 0 else 1
        if offset + n > data.size:
            raise ValueError(f"init-bin {bin_path}: ran out of data at {key} (need {n}, have {data.size - offset})")
        arr = data[offset:offset + n].reshape(shape).copy()
        sd[key] = torch.from_numpy(arr)
        offset += n
    if offset != data.size:
        raise ValueError(f"init-bin {bin_path}: size mismatch (consumed {offset}, file has {data.size})")
    model.load_state_dict(sd)


def write_bin(path, model, stack_encoder_dim=0, scalar_in_l1=False, no_fc_path=False, policy_head=False):
    """Write weights in float32 little-endian, in the fixed tensor order (tensor_order).
    tensors[0..7] ordering/format は既存モデルとの後方互換のため不変。"""
    import numpy as np

    sd = model.state_dict()
    order = tensor_order(stack_encoder_dim, scalar_in_l1, no_fc_path, policy_head)

    with open(path, "wb") as f:
        for key in order:
            arr = sd[key].detach().cpu().numpy().astype("<f4")
            f.write(np.ascontiguousarray(arr).tobytes())


def write_json(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_random_turn_context(rng, turn_context_dim, structured):
    """golden 用のターン文脈を生成する。structured=True なら実際の one-hot 構造
    （head 17 / phase 3 / dir 5 / steps 1）を持つベクトル、False なら一様乱数
    （数値パリティの頑健性確認用）。"""
    if not structured or turn_context_dim != 26:
        return [float(v) for v in rng.uniform(0.0, 1.0, size=turn_context_dim)]
    vec = [0.0] * 26
    vec[int(rng.randint(0, 17))] = 1.0          # head one-hot 17
    phase = int(rng.randint(0, 3))
    vec[17 + phase] = 1.0                        # phase one-hot 3
    vec[20 + (4 if phase == 0 else int(rng.randint(0, 4)))] = 1.0  # dir one-hot 5
    vec[25] = float(rng.randint(0, 17)) / 16.0   # steps scalar
    return vec


def build_golden(model, records, feature_dim, n_samples, seed, policy_head=False, turn_context_dim=26):
    """Sample n_samples records (including at least one with empty indices if present),
    compute forward output in eval/cpu/float32, and return the golden sample list."""
    import torch
    import numpy as np

    model_cpu = model.to("cpu")
    model_cpu.eval()

    rng = np.random.RandomState(seed)
    n = len(records)
    n_samples = min(n_samples, n)

    empty_idx_candidates = [i for i, r in enumerate(records) if len(r["indices"]) == 0]
    chosen = []
    if empty_idx_candidates:
        chosen.append(int(rng.choice(empty_idx_candidates)))

    remaining_pool = [i for i in range(n) if i not in chosen]
    rng.shuffle(remaining_pool)
    for i in remaining_pool:
        if len(chosen) >= n_samples:
            break
        chosen.append(i)

    samples = []
    with torch.no_grad():
        for k, i in enumerate(chosen):
            rec = records[i]
            x = build_dense_counts(rec["indices"], feature_dim)
            x_t = torch.tensor([x], dtype=torch.float32)
            s_t = torch.tensor([rec["scalars"]], dtype=torch.float32)
            if policy_head:
                # 多様な turnContext: 全ゼロ / 構造化 one-hot / 一様乱数 を混在させる
                if k % 3 == 0:
                    tc = [0.0] * turn_context_dim
                elif k % 3 == 1:
                    tc = make_random_turn_context(rng, turn_context_dim, structured=True)
                else:
                    tc = make_random_turn_context(rng, turn_context_dim, structured=False)
                tc_t = torch.tensor([tc], dtype=torch.float32)
                out, pol = model_cpu(x_t, s_t, tc_t)
                samples.append({
                    "indices": rec["indices"],
                    "scalars": rec["scalars"],
                    "turnContext": tc,
                    "out": float(out.item()),
                    "sigmoid": float(torch.sigmoid(out).item()),
                    "policyLogits": [float(v) for v in pol[0].tolist()],
                })
            else:
                out = model_cpu(x_t, s_t)
                samples.append({
                    "indices": rec["indices"],
                    "scalars": rec["scalars"],
                    "out": float(out.item()),
                    "sigmoid": float(torch.sigmoid(out).item()),
                })
    return samples


def main():
    args = parse_args()
    set_seed(args.seed)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: PyTorch is required to run train.py. Install via `pip install -r tools/nnue/requirements.txt`.", file=sys.stderr)
        sys.exit(1)

    print(f"[train] Loading data from {args.data} ...")
    t0 = time.time()
    records = load_records(args.data)
    print(f"[train] Loaded {len(records)} records in {time.time() - t0:.2f}s")

    board_size = records[0]["size"]
    for i, r in enumerate(records):
        if r["size"] != board_size:
            raise ValueError(f"Inconsistent board size at record {i}: {r['size']} != {board_size}")

    max_depth = 8
    scalar_dim = len(records[0]["scalars"])
    feature_dim = board_size * board_size * max_depth * 2

    print(f"[train] boardSize={board_size} featureDim={feature_dim} scalarDim={scalar_dim}")

    x_dense, scalars, y = build_dataset(records, feature_dim, scalar_dim, args.score_scale, args.lam, args.score_clip)

    n = len(records)
    train_idx, val_idx, split_by = split_by_pid(records, args.val_frac, args.seed)
    print(f"[train] Split ({split_by}): {len(train_idx)} train / {len(val_idx)} val")

    if split_by == "pid":
        num_train_positions = len({records[i]["pid"] for i in train_idx})
        num_val_positions = len({records[i]["pid"] for i in val_idx})
    else:
        num_train_positions = len(train_idx)
        num_val_positions = len(val_idx)

    device = resolve_device(args.device)

    if args.no_fc_path and args.stack_encoder_dim <= 0:
        print("ERROR: --no-fc-path には --stack-encoder-dim > 0 が必須です。", file=sys.stderr)
        sys.exit(1)

    if args.encoder_max_pool and args.stack_encoder_dim <= 0:
        print("ERROR: --encoder-max-pool には --stack-encoder-dim > 0 が必須です。", file=sys.stderr)
        sys.exit(1)

    model = build_model(feature_dim, scalar_dim, args.hidden1, args.hidden2, args.hidden3,
                         stack_encoder_dim=args.stack_encoder_dim, scalar_in_l1=args.scalar_in_l1,
                         no_fc_path=args.no_fc_path, encoder_max_pool=args.encoder_max_pool,
                         policy_head=args.policy_head, turn_context_dim=args.turn_context_dim,
                         num_actions=args.num_actions)

    if args.init_bin:
        load_bin_into(model, args.init_bin, stack_encoder_dim=args.stack_encoder_dim,
                      scalar_in_l1=args.scalar_in_l1, no_fc_path=args.no_fc_path,
                      policy_head=args.policy_head)
        print(f"[train] warm-start: loaded weights from {args.init_bin}")

    policy_targets = None
    policy_dist = None
    if args.policy_head:
        import numpy as np
        # ソフト目標（P4 自己対戦）優先。policyDist フィールドがあれば soft CE、
        # 無ければ policyTarget のハード CE にフォールバック（後方互換）。
        has_soft = any("policyDist" in r for r in records)
        if has_soft:
            policy_dist = np.zeros((n, args.num_actions), dtype=np.float32)
            n_with_soft = 0
            for i, r in enumerate(records):
                pd = r.get("policyDist")
                if pd is not None:
                    if len(pd) != args.num_actions:
                        raise ValueError(f"Record {i}: policyDist length {len(pd)} != {args.num_actions}")
                    policy_dist[i, :] = pd
                    n_with_soft += 1
            print(f"[train] policy head (SOFT target): {n_with_soft}/{n} records carry policyDist")
        else:
            policy_targets = np.array(
                [int(r["policyTarget"]) if "policyTarget" in r else -1 for r in records],
                dtype=np.int64)
            n_with_policy = int((policy_targets >= 0).sum())
            print(f"[train] policy head (HARD target): {n_with_policy}/{n} records carry policyTarget "
                  f"(CE is skipped for the rest)")

    turn_contexts = None
    if args.policy_head:
        import numpy as np
        turn_contexts = np.zeros((n, args.turn_context_dim), dtype=np.float32)
        for i, r in enumerate(records):
            tc = r.get("turnContext")
            if tc is not None:
                if len(tc) != args.turn_context_dim:
                    raise ValueError(f"Record {i}: turnContext length {len(tc)} != {args.turn_context_dim}")
                turn_contexts[i, :] = tc

    model, best_val_loss, last_train_loss, best_epoch, epochs_run = train_model(
        model, device, x_dense, scalars, y, train_idx, val_idx, args,
        policy_targets=policy_targets, turn_contexts=turn_contexts, policy_dist=policy_dist
    )
    if args.policy_head and getattr(model, "_best_val_top1", None) is not None:
        print(f"[train] best-epoch val policy top-1 = {model._best_val_top1:.4f}")

    print(f"[train] Done. bestEpoch={best_epoch} epochsRun={epochs_run} "
          f"bestValLoss={best_val_loss:.6f} lastTrainLoss={last_train_loss:.6f}")

    os.makedirs(args.out_dir, exist_ok=True)
    gen_tag = args.gen_tag if args.gen_tag else f"gen{args.generation:03d}"
    bin_path = os.path.join(args.out_dir, f"{gen_tag}.bin")
    json_path = os.path.join(args.out_dir, f"{gen_tag}.json")
    golden_path = os.path.join(args.out_dir, f"{gen_tag}.golden.json")

    write_bin(bin_path, model, stack_encoder_dim=args.stack_encoder_dim, scalar_in_l1=args.scalar_in_l1,
              no_fc_path=args.no_fc_path, policy_head=args.policy_head)
    print(f"[train] Wrote weights: {bin_path}")

    CELL_SLOT_DIM = 16
    h1_contrib = 0 if args.no_fc_path else args.hidden1
    enc_contrib = args.stack_encoder_dim * (2 if (args.stack_encoder_dim > 0 and args.encoder_max_pool) else 1)
    tc_contrib = args.turn_context_dim if args.policy_head else 0
    h2_input_dim = h1_contrib + scalar_dim + enc_contrib + tc_contrib

    if args.no_fc_path:
        tensors = [
            {"name": "W2", "shape": [args.hidden2, h2_input_dim]},
            {"name": "b2", "shape": [args.hidden2]},
            {"name": "W3", "shape": [args.hidden3, args.hidden2]},
            {"name": "b3", "shape": [args.hidden3]},
            {"name": "W4", "shape": [1, args.hidden3]},
            {"name": "b4", "shape": [1]},
        ]
    else:
        tensors = [
            {"name": "W1", "shape": [args.hidden1, feature_dim]},
            {"name": "b1", "shape": [args.hidden1]},
            {"name": "W2", "shape": [args.hidden2, h2_input_dim]},
            {"name": "b2", "shape": [args.hidden2]},
            {"name": "W3", "shape": [args.hidden3, args.hidden2]},
            {"name": "b3", "shape": [args.hidden3]},
            {"name": "W4", "shape": [1, args.hidden3]},
            {"name": "b4", "shape": [1]},
        ]
    if args.stack_encoder_dim > 0:
        tensors += [
            {"name": "Wenc", "shape": [args.stack_encoder_dim, CELL_SLOT_DIM]},
            {"name": "benc", "shape": [args.stack_encoder_dim]},
        ]
    if args.scalar_in_l1 and not args.no_fc_path:
        tensors += [
            {"name": "Ws1", "shape": [args.hidden1, scalar_dim]},
        ]
    if args.policy_head:
        tensors += [
            {"name": "Wp", "shape": [args.num_actions, args.hidden3]},
            {"name": "bp", "shape": [args.num_actions]},
        ]

    meta = {
        "format": "nnue-ukeja-v1",
        "generation": args.generation,
        "boardSize": board_size,
        "featureDim": feature_dim,
        "scalarDim": scalar_dim,
        "maxDepth": max_depth,
        "hidden": [args.hidden1, args.hidden2, args.hidden3],
        "concatScalarsAfterLayer": 1,
        "activation": "clipped_relu",
        "clipMin": 0.0,
        "clipMax": 1.0,
        "scoreScale": args.score_scale,
        "scoreClip": args.score_clip,
        "lambda": args.lam,
        "dtype": "float32",
        "byteOrder": "little-endian",
        "tensors": tensors,
        "train": {
            "dataFile": args.data,
            "numRecords": n,
            "valLoss": best_val_loss,
            "trainLoss": last_train_loss,
            "epochsRun": epochs_run,
            "bestEpoch": best_epoch,
            "device": str(device),
            "seed": args.seed,
            "splitBy": split_by,
            "numTrainPositions": num_train_positions,
            "numValPositions": num_val_positions,
            "valPolicyTop1": getattr(model, "_best_val_top1", None),
        },
    }
    if args.stack_encoder_dim > 0:
        meta["stackEncoder"] = {"dim": args.stack_encoder_dim, "cellSlotDim": CELL_SLOT_DIM}
        if args.encoder_max_pool:
            meta["stackEncoder"]["maxPool"] = True
    if args.scalar_in_l1 and not args.no_fc_path:
        meta["scalarInL1"] = True
    if args.no_fc_path:
        meta["noFcPath"] = True
    if args.policy_head:
        meta["policyHead"] = {"actions": args.num_actions}
        meta["turnContextDim"] = args.turn_context_dim

    write_json(json_path, meta)
    print(f"[train] Wrote metadata: {json_path}")

    golden_samples = build_golden(model, records, feature_dim, args.golden_samples, args.seed,
                                  policy_head=args.policy_head, turn_context_dim=args.turn_context_dim)
    golden = {
        "model": gen_tag,
        "scoreScale": args.score_scale,
        "samples": golden_samples,
    }
    write_json(golden_path, golden)
    print(f"[train] Wrote golden samples ({len(golden_samples)}): {golden_path}")

    print("[train] Summary: "
          f"records={n} trainN={len(train_idx)} valN={len(val_idx)} "
          f"bestValLoss={best_val_loss:.6f} trainLoss={last_train_loss:.6f} "
          f"bestEpoch={best_epoch} epochsRun={epochs_run} device={device}")


if __name__ == "__main__":
    main()

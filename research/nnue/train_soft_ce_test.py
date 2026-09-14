"""train_soft_ce_test.py — P4: soft_cross_entropy の単体テスト。

one-hot 目標のとき通常の CrossEntropyLoss(index) と一致すること、および
勾配方向の健全性（目標アクションのロジットを上げると損失が下がる）を検証する。
"""
import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


def _load_train():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("train_mod", os.path.join(here, "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train()


def test_soft_ce_matches_hard_ce_on_one_hot():
    """π が one-hot のとき soft CE == nn.CrossEntropyLoss(index=argmax)。"""
    torch.manual_seed(0)
    B, A = 8, 81
    logits = torch.randn(B, A, requires_grad=True)
    idx = torch.randint(0, A, (B,))
    onehot = torch.zeros(B, A)
    onehot[torch.arange(B), idx] = 1.0

    soft = train.soft_cross_entropy(logits, onehot)
    hard = nn.CrossEntropyLoss()(logits, idx)
    assert torch.allclose(soft, hard, atol=1e-6), f"soft={soft.item()} hard={hard.item()}"


def test_soft_ce_gradient_direction():
    """目標分布に確率質量がある側のロジットを上げると損失が単調に下がる（健全な勾配方向）。"""
    A = 81
    target = torch.zeros(1, A)
    target[0, 5] = 0.7
    target[0, 10] = 0.3

    base = torch.zeros(1, A)
    losses = []
    for boost in [0.0, 1.0, 2.0, 4.0]:
        logits = base.clone()
        logits[0, 5] = boost
        logits[0, 10] = boost  # 目標側を持ち上げる
        losses.append(train.soft_cross_entropy(logits, target).item())
    # 目標側ロジットを上げるほど損失は減少する
    for i in range(len(losses) - 1):
        assert losses[i + 1] < losses[i], f"loss should decrease: {losses}"


def test_soft_ce_ignores_zero_mass_rows():
    """行和0（教師なし）の行は損失に寄与しない。"""
    A = 81
    logits = torch.randn(2, A)
    target = torch.zeros(2, A)
    target[0, 3] = 1.0  # 行0のみ教師あり
    # 行1は全ゼロ（教師なし）
    loss_both = train.soft_cross_entropy(logits, target)
    loss_only0 = train.soft_cross_entropy(logits[:1], target[:1])
    assert torch.allclose(loss_both, loss_only0, atol=1e-6)


def test_soft_ce_all_zero_returns_zero_grad_scalar():
    """全行が教師なしなら勾配ゼロのスカラー（NaN を出さない）。"""
    A = 81
    logits = torch.randn(3, A, requires_grad=True)
    target = torch.zeros(3, A)
    loss = train.soft_cross_entropy(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))

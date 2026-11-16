"""Guided refinement of the NIW head (niw_refine="feat", NIWRefine; docs/finetune_plan.md Phase 3, "B").

A checkpoint without the refinement must load with only niw_refine.* missing and reproduce the head exactly
(zero-initialised output conv); the feature tap must be the geometry DPT's penultimate features; d_s must rescale
L' (so Psi by exp(2 d_s)) without changing its orientation; with the NIW head and the gated residual frozen,
gradients must reach the refinement only.
Run with: python -m pytest tests/  (or, without pytest: python tests/test_niw_refine.py)
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from test_niw_zxy import predict, tiny_model  # noqa: E402

BASE = 'catmlp+dpt+xyz_niw_dpt+residual_gated:niw_order="zxy"'
REFINE = 'catmlp+dpt+xyz_niw_dpt+residual_gated:niw_order="zxy",niw_refine="feat"'
NIW_KEYS = ("xyz_niw_kappa", "xyz_niw_nu", "xyz_niw_Lp", "xyz_niw_Lmat", "xyz_niw_psi_diag", "xyz_niw_var_diag")


def refined_from(base):
    net = tiny_model(REFINE)
    missing, unexpected = net.load_state_dict(base.state_dict(), strict=False)
    return net, missing, unexpected


def test_base_checkpoint_loads_with_only_refine_missing():
    _, missing, unexpected = refined_from(tiny_model(BASE))
    assert not unexpected, unexpected
    assert missing and all(".niw_refine." in k for k in missing), missing


def test_zero_init_reproduces_base_head():
    base = tiny_model(BASE)
    net, _, _ = refined_from(base)
    a, b = predict(base), predict(net)
    for k in NIW_KEYS + ("pts3d", "conf"):
        assert torch.equal(a[k], b[k]), k
    assert torch.equal(b["xyz_niw_refine"], torch.zeros_like(b["xyz_niw_refine"]))


def test_feature_tap_is_penultimate():
    """dpt.head[4] applied to the tapped features gives the geometry DPT's own output."""
    head = tiny_model(REFINE).downstream_head1
    path_1 = torch.randn(2, head.dpt.head[0].in_channels, 5, 6)
    with torch.no_grad():
        full = head.dpt.head(path_1)
        feat = head._geometry_features(path_1)
        assert feat.shape[1] == head.niw_refine.proj.in_channels
        assert torch.allclose(head.dpt.head[4](feat), full)
        assert (feat >= 0).all()  # after the ReLU


def test_scale_rescales_psi_not_orientation():
    """Only d_s nonzero: every L' entry, so Lmat, is multiplied by exp(d_s); kappa and nu are unchanged."""
    base = tiny_model(BASE)
    net, _, _ = refined_from(base)
    for head in (net.downstream_head1, net.downstream_head2):
        torch.nn.init.normal_(head.niw_refine.out.weight[2:3], std=0.3)
    a, b = predict(base), predict(net)
    s = torch.exp(b["xyz_niw_refine"][:, 2:3])
    assert (b["xyz_niw_refine"][:, 2:3].abs() > 1e-4).any()
    assert torch.allclose(b["xyz_niw_Lp"], a["xyz_niw_Lp"] * s, rtol=1e-5, atol=1e-6)
    assert torch.allclose(b["xyz_niw_Lmat"], a["xyz_niw_Lmat"] * s.unsqueeze(1), rtol=1e-5, atol=1e-6)
    for k in ("xyz_niw_kappa", "xyz_niw_nu"):
        assert torch.equal(a[k], b[k]), k


def test_kappa_nu_corrections_apply_to_raw_values():
    base = tiny_model(BASE)
    net, _, _ = refined_from(base)
    for head in (net.downstream_head1, net.downstream_head2):
        torch.nn.init.normal_(head.niw_refine.out.weight[0:2], std=0.3)
    a, b = predict(base), predict(net)
    sp_inv = lambda y: y + torch.log(-torch.expm1(-y))  # noqa: E731  inverse softplus
    eps = net.downstream_head1.eps
    d_kappa, d_nu = b["xyz_niw_refine"][:, 0:1], b["xyz_niw_refine"][:, 1:2]
    k = torch.nn.functional.softplus(sp_inv((a["xyz_niw_kappa"] - eps).double()) + d_kappa.double()) + eps
    n = 4.0 + torch.nn.functional.softplus(sp_inv((a["xyz_niw_nu"] - 4.0 - eps).double()) + d_nu.double()) + eps
    assert torch.allclose(b["xyz_niw_kappa"].double(), k, rtol=1e-4, atol=1e-5)
    assert torch.allclose(b["xyz_niw_nu"].double(), n, rtol=1e-4, atol=1e-5)
    assert torch.equal(a["xyz_niw_Lp"], b["xyz_niw_Lp"])  # d_s = 0


def test_gradients_reach_only_refinement_when_head_frozen():
    """Option (a): everything but niw_refine frozen. A loss on the NIW outputs must give the refinement's output
    conv a gradient (its other layers get none yet: the output conv is zero) and nothing else."""
    net, _, _ = refined_from(tiny_model(BASE))
    for name, p in net.named_parameters():
        p.requires_grad = ".niw_refine." in name
    g = torch.Generator().manual_seed(2)
    views = [dict(img=torch.randn(1, 3, 32, 32, generator=g), true_shape=torch.tensor([[32, 32]]), instance=[str(i)])
             for i in range(2)]
    pred1, pred2 = net(*views)
    loss = sum((p["xyz_niw_kappa"].log() + p["xyz_niw_nu"].log() + p["xyz_niw_Lp"].abs().log()).mean()
               for p in (pred1, pred2))
    loss.backward()
    for h in ("downstream_head1", "downstream_head2"):
        out = getattr(net, h).niw_refine.out
        assert out.weight.grad is not None and out.weight.grad.abs().sum() > 0, h
    assert all(p.grad is None for n, p in net.named_parameters() if ".niw_refine." not in n)


def test_input_statistics_set_once_from_first_training_batch():
    net, _, _ = refined_from(tiny_model(BASE))
    ref = net.downstream_head1.niw_refine
    predict(net)  # eval: statistics stay at identity
    assert not bool(ref.in_ready) and torch.equal(ref.in_std, torch.ones_like(ref.in_std))
    net.train()
    predict(net)
    assert bool(ref.in_ready) and not torch.equal(ref.in_std, torch.ones_like(ref.in_std))
    mean, std = ref.in_mean.clone(), ref.in_std.clone()
    g = torch.Generator().manual_seed(7)
    views = [dict(img=torch.randn(1, 3, 32, 32, generator=g), true_shape=torch.tensor([[32, 32]]), instance=[str(i)])
             for i in range(2)]
    with torch.no_grad():
        net(*views)  # a different batch must not change them
    assert torch.equal(mean, ref.in_mean) and torch.equal(std, ref.in_std)
    assert all(k in net.state_dict() for k in ("downstream_head1.niw_refine.in_mean",
                                               "downstream_head1.niw_refine.in_std",
                                               "downstream_head1.niw_refine.in_ready"))


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)

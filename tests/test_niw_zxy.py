"""Sign-consistent NIW Cholesky factor (niw_order="zxy", docs/seam_mechanism.md section 6).

The head predicts L' with L' L'^T = P Psi P^T, P (x, y, z) = (z, x, y). Training and eval NLL permute the residual
instead of converting L' back, so they must equal the Student-t NLL of the dense Psi in xyz; the trace (u_epi)
and log det must not change; the per-axis outputs must come back in xyz.
Run with: python -m pytest tests/  (or, without pytest: python tests/test_niw_zxy.py)
"""
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "eval")]

from mast3r.catmlp_dpt_head import NIW_XYZ_TO_ZXY, NIW_ZXY_TO_XYZ  # noqa: E402
from mast3r.losses_evidential import evidential_niw_loss_3d  # noqa: E402
from uq_eval_utils import niw_nll_3d  # noqa: E402
from mast3r.model import AsymmetricMASt3R  # noqa: E402

inf = float("inf")

P = torch.zeros(3, 3, dtype=torch.float64)
P[0, 2] = P[1, 0] = P[2, 1] = 1.0  # P @ (x, y, z) = (z, x, y)


def tiny_model(head_type="catmlp+dpt+xyz_niw_dpt+residual_gated"):
    """Same tiny model as test_post_smoothing.py (kept separate so this file runs without pytest)."""
    torch.manual_seed(0)
    return AsymmetricMASt3R(
        pos_embed="RoPE100", patch_embed_cls="PatchEmbedDust3R", img_size=(32, 32),
        head_type=head_type, output_mode="pts3d+desc24",
        depth_mode=("exp", -inf, inf), conf_mode=("exp", 1, inf),
        enc_embed_dim=64, enc_depth=2, enc_num_heads=4, dec_embed_dim=64, dec_depth=10, dec_num_heads=4,
        two_confs=True, desc_conf_mode=("exp", 0, inf), landscape_only=False).eval()


def predict(net):
    g = torch.Generator().manual_seed(2)
    views = [dict(img=torch.randn(1, 3, 32, 32, generator=g), true_shape=torch.tensor([[32, 32]]), instance=[str(i)])
             for i in range(2)]
    with torch.no_grad():
        pred1, _ = net(*views)
    return pred1


def random_niw(B=2, H=3, W=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)  # noqa: E731
    L = r(B, 6, H, W)
    for c in (0, 2, 5):  # positive diagonal, as after the head's softplus
        L[:, c] = torch.nn.functional.softplus(L[:, c]) + 0.1
    kappa = torch.nn.functional.softplus(r(B, 1, H, W)) + 0.1
    nu = 4.0 + torch.nn.functional.softplus(r(B, 1, H, W))
    return r(B, 3, H, W), r(B, 3, H, W), kappa, nu, L


def lmat(L):
    """[B,6,H,W] -> [B,H,W,3,3] lower-triangular."""
    B, _, H, W = L.shape
    M = torch.zeros(B, H, W, 3, 3, dtype=L.dtype)
    for c, (i, j) in enumerate(((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2))):
        M[..., i, j] = L[:, c]
    return M


def dense_nll(y, m, kappa, nu, Psi):
    """Multivariate Student-t NLL of the NIW predictive, from a dense Psi in xyz ([B,H,W,3,3])."""
    d = 3
    r = (y - m).permute(0, 2, 3, 1).unsqueeze(-1)
    k, n = kappa[:, 0], nu[:, 0]
    nu_p = n - (d - 1)
    scale = (k + 1) / (k * nu_p)
    Sigma = Psi * scale[..., None, None]
    delta = (r.transpose(-1, -2) @ torch.linalg.solve(Sigma, r))[..., 0, 0]
    return -(torch.lgamma(0.5 * (nu_p + d)) - torch.lgamma(0.5 * nu_p)
             - 0.5 * (d * torch.log(nu_p * math.pi) + torch.logdet(Sigma))
             - 0.5 * (nu_p + d) * torch.log1p(delta / nu_p))


def test_permutation_tables():
    x = torch.tensor([10.0, 20.0, 30.0])  # x, y, z
    assert x[list(NIW_XYZ_TO_ZXY)].tolist() == [30.0, 10.0, 20.0]
    assert x[list(NIW_XYZ_TO_ZXY)][list(NIW_ZXY_TO_XYZ)].tolist() == x.tolist()
    assert torch.equal(P @ x.double(), x.double()[list(NIW_XYZ_TO_ZXY)])


def test_training_and_eval_nll_match_dense_psi():
    y, m, kappa, nu, L = random_niw()
    Lm = lmat(L)
    for order, Psi in (("xyz", Lm @ Lm.transpose(-1, -2)),
                       ("zxy", P.T @ (Lm @ Lm.transpose(-1, -2)) @ P)):
        ref = dense_nll(y, m, kappa, nu, Psi)
        _, stats = evidential_niw_loss_3d(y, m, kappa, nu, L, lambda_evi=0.0, eps=1e-12, order=order)
        assert math.isclose(stats["nll"], float(ref.mean()), rel_tol=1e-9, abs_tol=1e-9), (order, stats)
        ev = niw_nll_3d(y, m, kappa, nu, L, None, eps=1e-12, order=order)
        assert math.isclose(ev, float(ref.mean()), rel_tol=1e-9, abs_tol=1e-9), order


def test_zxy_differs_from_xyz():
    """Guards against the order being silently ignored."""
    y, m, kappa, nu, L = random_niw(seed=3)
    a = niw_nll_3d(y, m, kappa, nu, L, None, order="xyz")
    b = niw_nll_3d(y, m, kappa, nu, L, None, order="zxy")
    assert abs(a - b) > 1e-3


def test_trace_and_logdet_invariant():
    _, _, _, _, L = random_niw(seed=1)
    Lm = lmat(L)
    Psi_zxy = Lm @ Lm.transpose(-1, -2)
    Psi_xyz = P.T @ Psi_zxy @ P
    assert torch.allclose(torch.diagonal(Psi_xyz, dim1=-2, dim2=-1).sum(-1),
                          torch.diagonal(Psi_zxy, dim1=-2, dim2=-1).sum(-1))
    assert torch.allclose(torch.logdet(Psi_xyz), 2 * torch.log(L[:, [0, 2, 5]]).sum(1))


def test_head_outputs():
    """zxy head: Lmat is a square root of Psi in xyz, psi_diag is xyz-ordered, and the order is reported."""
    net = tiny_model('catmlp+dpt+xyz_niw_dpt+residual_gated:niw_order="zxy"')
    for head in (net.downstream_head1, net.downstream_head2):
        assert head.niw_order == "zxy"
        torch.nn.init.normal_(head.dpt_xyz_niw.head[-1].weight, std=0.1)  # non-trivial L
    pred = predict(net)
    assert pred["xyz_niw_order"] == "zxy"
    Lp = pred["xyz_niw_Lp"].double()
    Lz = lmat(Lp)                                                   # L' (rows z, x, y)
    Psi_xyz = P.T @ (Lz @ Lz.transpose(-1, -2)) @ P
    S = pred["xyz_niw_Lmat"].double().permute(0, 3, 4, 1, 2)          # [B,H,W,3,3]
    assert torch.allclose(S @ S.transpose(-1, -2), Psi_xyz, atol=1e-5)
    diag = pred["xyz_niw_psi_diag"].double().permute(0, 2, 3, 1)
    assert torch.allclose(diag, torch.diagonal(Psi_xyz, dim1=-2, dim2=-1), atol=1e-5)

    xyz = predict(tiny_model())
    assert xyz["xyz_niw_order"] == "xyz"


def test_released_checkpoint_loads_into_zxy_head():
    """Same parameter names and shapes: a released-layout state dict loads with no missing/unexpected keys."""
    sd = tiny_model().state_dict()
    missing, unexpected = tiny_model('catmlp+dpt+xyz_niw_dpt+residual_gated:niw_order="zxy"').load_state_dict(
        sd, strict=False)
    assert not missing and not unexpected


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)

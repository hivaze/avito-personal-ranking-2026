"""Mechanism diagnostics for the nnx attn experiments (A1-A5), computed on a fixed random subset of val feeds.

Used inside train.py at every eval, and standalone on any checkpoint (e.g. the shared baseline):
  CUDA_VISIBLE_DEVICES=N python diag.py --ckpt <repo>/nn_x/optim/ckpt/base_s23.pt
"""
import argparse
import json
import math

import numpy as np
import torch

from model import Attn, HistRetrieval, MoEFF, N_CAT

M_BUCKETS = [(0, 32), (33, 64), (65, 128), (129, 400)]


def _mean(x, m):
    m = m.float()
    return float((x.float() * m).sum() / m.sum().clamp_min(1))


@torch.no_grad()
def run_diag(model, data, n_feeds=3000, seed=0, cand_budget=16384):
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    fids = np.sort(rng.choice(data.n, min(n_feeds, data.n), replace=False))
    attn = {n: m for n, m in model.named_modules() if isinstance(m, Attn)}
    moes = {n: m for n, m in model.named_modules() if isinstance(m, MoEFF)}
    retr = [m for m in model.modules() if isinstance(m, HistRetrieval)]
    for m in list(attn.values()) + list(moes.values()) + retr:
        m.record = True
    model.record = True
    acc = {}

    def add(key, val, w):
        s, n = acc.get(key, (0.0, 0.0))
        acc[key] = (s + val * w, n + w)

    moe_tab = {n: torch.zeros(N_CAT, m.n, dtype=torch.float64) for n, m in moes.items()}
    for fb in data.batches(fids, cand_budget, 1024, shuffle=False):
        b = data.gather(fb)
        model.stats = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(b, use_pos=False)
        cm = b["cmask"]
        nq = float(cm.sum())
        hq = torch.cat([torch.ones_like(b["hmask"][:, :1]), b["hmask"]], 1)
        for name, m in attn.items():
            st = m.stats
            if not st:
                continue
            qm = hq if name.startswith("hist_enc") else cm  # [B,N] valid queries
            qmh = qm[:, None, :].expand_as(st["ent"])
            w = float(qm.sum())
            add(f"{name}/ent", _mean(st["ent"], qmh), w)
            add(f"{name}/pmax", _mean(st["pmax"], qmh), w)
            if st["gate"] is not None:
                add(f"{name}/gate", _mean(st["gate"], qmh), w)
            if name.startswith("cross"):
                add(f"{name}/user_mass", _mean(st["p0"], qmh), w)
                for hh in range(st["p0"].shape[1]):
                    add(f"{name}/user_mass_h{hh}", _mean(st["p0"][:, hh], qm), w)
                if st["gate"] is not None:
                    add(f"{name}/user_mass_gated", _mean(st["p0"] * st["gate"], qmh), w)
                    # candidates of feeds without history: is the read switched off?
                    nh = (b["H"] == 0)[:, None].expand_as(qm) & qm
                    if nh.any():
                        add(f"{name}/gate_noHist", _mean(st["gate"], nh[:, None, :].expand_as(st["gate"])),
                            float(nh.sum()))
            if name.startswith("sets.0"):
                Mf = b["M"]
                for lo, hi in M_BUCKETS:
                    fm = ((Mf >= lo) & (Mf <= hi))[:, None] & cm
                    if fm.any():
                        e = st["ent"].mean(1)  # mean over heads [B,N]
                        add(f"{name}/ent_M{lo}-{hi}", _mean(e, fm), float(fm.sum()))
                        lm = torch.log(Mf.float().clamp_min(2))[:, None].expand_as(e)
                        add(f"{name}/entnorm_M{lo}-{hi}", _mean(e / lm, fm), float(fm.sum()))
            m.stats = {}
        for name, m in moes.items():
            st = m.stats
            if not st:
                continue
            sel = st["sel"][cm]  # [T,k]
            cat = b["ccat"][cm]
            for j in range(sel.shape[1]):
                moe_tab[name].index_put_((cat.cpu(), sel[:, j].cpu()), torch.ones(len(cat), dtype=torch.float64),
                                         accumulate=True)
            m.stats = {}
        for m in retr:
            st = m.stats
            if not st:
                continue
            add("retr/sink", _mean(st["sink"], cm), nq)
            vm = cm & st["valid"]
            add("retr/sink_hasHist", _mean(st["sink"], vm), float(vm.sum()))
            lo, hi = vm & (st["smax"] < 0.5), vm & (st["smax"] > 0.9)
            if lo.any():
                add("retr/sink_smax<0.5", _mean(st["sink"], lo), float(lo.sum()))
            if hi.any():
                add("retr/sink_smax>0.9", _mean(st["sink"], hi), float(hi.sum()))
            add("retr/out_norm", _mean(st["out_norm"], cm), nq)
            m.stats = {}
        if "retr_ratio" in model.stats:
            add("retr/ratio", _mean(model.stats["retr_ratio"], cm), nq)
        if "fox_D" in model.stats:
            H = b["H"]
            for i, D in enumerate(model.stats["fox_D"]):
                D = D[:, :, 0, 1:]  # [B,h,K] (drop the user token)
                Kx = D.shape[-1]
                hv = (torch.arange(Kx, device=D.device)[None, :] < H[:, None])[:, None, :]
                below = (D < -1.0) & hv
                first = torch.where(below, torch.arange(Kx, device=D.device), Kx).min(-1).values.float()  # [B,h]
                horizon = torch.minimum(first, H[:, None].float())
                fm = (H > 0)[:, None].expand_as(horizon)
                for hh in range(D.shape[1]):
                    add(f"fox{i}/horizon_h{hh}", _mean(horizon[:, hh], fm[:, hh]), float(fm[:, hh].sum()))
                if Kx > 32:
                    sel = H > 32
                    if sel.any():
                        acc.setdefault(f"fox{i}/D32_list", []).append(D[sel][:, :, 32].float().cpu())
    out = {k: v[0] / max(v[1], 1e-9) for k, v in acc.items() if isinstance(v, tuple)}
    for k, v in acc.items():
        if isinstance(v, list):
            d = torch.cat(v)  # [n,h]
            out[k.replace("_list", "_mean")] = [round(float(x), 3) for x in d.mean(0)]
            out[k.replace("_list", "_std")] = [round(float(x), 3) for x in d.std(0)]
    for name, tab in moe_tab.items():
        p = tab / tab.sum().clamp_min(1)
        pe, pc = p.sum(0), p.sum(1)
        ent = float(-(pe * torch.log(pe.clamp_min(1e-12))).sum())
        nz = p > 0
        mi = float((p[nz] * torch.log(p[nz] / (pc[:, None] * pe[None, :])[nz])).sum())
        out[f"{name}/load_ent"] = ent / math.log(len(pe))
        out[f"{name}/load_maxmin"] = float(pe.max() / pe.min().clamp_min(1e-9))
        out[f"{name}/mi_cat"] = mi
        out[f"{name}/mi_cat_norm"] = mi / max(ent, 1e-9)
    # parameter-level diagnostics
    for name, m in attn.items():
        if getattr(m, "ssmax", False):
            out[f"{name}/ss_x_logref"] = [round(float(x), 3) for x in m.ss.detach().cpu()]
        if getattr(m, "qk_norm", False):
            out[f"{name}/gq_norm"] = float(m.qn.weight.norm())
            out[f"{name}/gk_norm"] = float(m.kn.weight.norm())
        if hasattr(m, "bias_scale"):
            out[f"{name}/sim_a"] = [round(float(x), 3) for x in m.bias_scale.detach().cpu()]
    for m in retr:
        out["retr/a"] = float(m.a)
        out["retr/sink_logit"] = float(m.sink)
    for m in list(attn.values()) + list(moes.values()) + retr:
        m.record = False
    model.record = False
    model.stats = {}
    if was_training:
        model.train()
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


if __name__ == "__main__":
    from data import SplitData
    from model import Ranker
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_feeds", type=int, default=3000)
    a = p.parse_args()
    dev = torch.device("cuda:0")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = Ranker(**ck["cfg"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    va = SplitData("val", dev, max_hist=ck["args"]["max_hist"])
    print(json.dumps(run_diag(model, va, a.n_feeds)))

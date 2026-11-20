"""Brute-force (per pseudo-feed, python loops, numpy) checker of a pseudo batch built by cd_arms11.build_pseudo_batch from its source
batch b. Independent of the vectorised builder: re-derives every quantity from the source row and compares. Returns the list of
violation strings (empty = every pseudo-feed satisfies the construction)."""
import numpy as np
import torch


def check_pseudo_batch(b, bp, cemb, src_ok, gmin, gmax, tmax):
    v = []
    P = bp["y"].shape[0]
    het, hes, hv, hidx, hmask = (b[k].cpu().numpy() for k in ("het", "hes", "hemb_ok", "hidx", "hmask"))
    cmask, y = b["cmask"].cpu().numpy(), b["y"].cpu().numpy()
    cemb_np, src_ok_np = cemb.cpu().numpy(), src_ok.cpu().numpy()
    Hc = b["H"].cpu().numpy()
    he_raw, ce_raw = b["he_raw"], b["ce_raw"]
    src, t_all, g_all = (bp[k].cpu().numpy() for k in ("pseudo_src", "pseudo_t", "pseudo_g"))
    valid_rows = bp["pseudo_valid"].cpu().numpy()
    for p in range(P):
        i, t, g = int(src[p]), int(t_all[p]), int(g_all[p])
        tag = f"p{p} src{i} t{t} g{g}"
        if not valid_rows[p]:  # padded row (a dropped draw): must carry no positive and duplicate a valid row's source
            if bp["y"][p].abs().sum() > 0:
                v.append(f"{tag}: padded row has a positive")
            if not valid_rows[p % max(1, int(valid_rows.sum()))]:
                v.append(f"{tag}: padded row does not map to a valid row")
            continue
        if not src_ok_np[i]:
            v.append(f"{tag}: source row blocked (holdout user)")
        if not (0 <= t <= tmax):
            v.append(f"{tag}: t out of range")
        if not (gmin <= g <= gmax):
            v.append(f"{tag}: g out of range")
        if not (hmask[i, t] and hv[i, t] and het[i, t] == 1 and hes[i, t] == 1):
            v.append(f"{tag}: target not a valid type-1/source-1 event with an embedding")
        older = np.arange(hidx.shape[1]) > t
        if ((hidx[i] == hidx[i, t]) & hv[i] & older).any():
            v.append(f"{tag}: target item occurs in an older event (not new)")
        if ((cemb_np[i] == hidx[i, t]) & cmask[i]).any():
            v.append(f"{tag}: target item is a candidate of the source feed")
        H = int(Hc[i])
        Hp = H - t - g
        if Hp < 1 or int(bp["H"][p]) != Hp:
            v.append(f"{tag}: H' {int(bp['H'][p])} != H_cap - t - g = {Hp} (or < 1)")
            continue
        Kp = bp["hmask"].shape[1]
        if not np.array_equal(bp["hmask"][p].cpu().numpy(), np.arange(Kp) < Hp):
            v.append(f"{tag}: hmask' != arange < H'")
        if not torch.equal(bp["he_raw"][p, :Hp], he_raw[i, t + g:H]):
            v.append(f"{tag}: history embeddings != source events at index >= t + g")
        for k_ in ("hcat", "het", "hes"):
            if not torch.equal(bp[k_][p, :Hp], b[k_][i, t + g:H]):
                v.append(f"{tag}: {k_}' != source {k_} at index >= t + g")
        if not torch.equal(bp["hemb_ok"][p, :Hp], b["hemb_ok"][i, t + g:H]):
            v.append(f"{tag}: hemb_ok' != source")
        if Hp < Kp and (bp["hemb_ok"][p, Hp:].any() or bp["he_raw"][p, Hp:].abs().sum() > 0):
            v.append(f"{tag}: padded history not blank")
        # candidates
        neg = cmask[i] & (y[i] == 0)
        fut = ((cemb_np[i][:, None] == hidx[i][None, :t + g]) & hv[i][None, :t + g]).any(1)
        neg = neg & ~fut
        n = int(neg.sum())
        if n < 1:
            v.append(f"{tag}: no negative")
        if int(bp["M"][p]) != 1 + n:
            v.append(f"{tag}: M' {int(bp['M'][p])} != 1 + n_neg {1 + n}")
            continue
        Mx = bp["cmask"].shape[1]
        if not np.array_equal(bp["cmask"][p].cpu().numpy(), np.arange(Mx) < 1 + n):
            v.append(f"{tag}: cmask' != arange < M'")
        if not torch.equal(bp["ce_raw"][p, 0], he_raw[i, t]) or bool(bp["cmiss"][p, 0]) or int(bp["ccat"][p, 0]) != int(b["hcat"][i, t]):
            v.append(f"{tag}: slot 0 != the target history event")
        if not torch.equal(bp["ce_raw"][p, 1:1 + n], ce_raw[i][torch.from_numpy(neg)]):
            v.append(f"{tag}: negatives' embeddings != the source feed's non-clicked candidates in order")
        if not torch.equal(bp["ccat"][p, 1:1 + n], b["ccat"][i][torch.from_numpy(neg)]):
            v.append(f"{tag}: negatives' categories != source")
        if not torch.equal(bp["cmiss"][p, 1:1 + n], b["cmiss"][i][torch.from_numpy(neg)]):
            v.append(f"{tag}: negatives' cmiss != source")
        pos_items = set(cemb_np[i][cmask[i] & (y[i] > 0)].tolist()) - {-1}  # no clicked item of the source feed among the negatives
        if pos_items & set(cemb_np[i][neg].tolist()):
            v.append(f"{tag}: a clicked candidate's item of the source feed appears among the negatives")
        yp = bp["y"][p].cpu().numpy()
        if not (yp[0] == 1 and yp.sum() == 1):
            v.append(f"{tag}: y' is not one-hot on slot 0")
        if bp["ue_raw"][p].abs().sum() > 0:
            v.append(f"{tag}: user_emb not zero")
        if bp["cf"][p].abs().sum() > 0 or ("xt" in bp and bp["xt"][p].abs().sum() > 0):
            v.append(f"{tag}: f1-f5 / xt not zero")
        if "dom" in bp and bp["dom"][p].tolist() != [0.0, 0.0, 1.0]:
            v.append(f"{tag}: domain one-hot != row 2")
        if int(bp["user0"][p]) != int(b["user0"][i]) or int(bp["fidx"][p]) != int(b["fidx"][i]):
            v.append(f"{tag}: user0 / fidx not the source row's")
    return v

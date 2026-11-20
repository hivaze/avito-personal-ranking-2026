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


# ---------------------------------------------------------------------------------------------------------------------------------
# pseudokd port (2026-11-17): locality / "built like a real feed" checks added on top of the verbatim cd_arms11 checker above.
REAL_ONLY = {"ci", "pos", "ce", "he", "un"}  # gathered-batch keys a pseudo batch does not carry (the pseudo forward reads none of them)
PSEUDO_KEYS = {"pseudo_src", "pseudo_t", "pseudo_g", "pseudo_valid"}


def check_pseudo_extra(b, bp, max_hist=256):
    """(1) key set == the real batch's keys minus REAL_ONLY plus PSEUDO_KEYS, same dtypes, same trailing dims;
    (2) every pseudo row is a function of ONE source row of b: hidx' == the source's hidx at index >= t + g (the user's own stored
        history, strictly older than the target and the gap; -1 on padding), the target slot is the source's event t, user0 / fidx are the
        source's (cd_arms11 checker), and no index < t + g of the source history reaches the pseudo history;
    (3) padding written as gather() writes it: candidate slots >= M' blank (ce_raw 0, cmiss True, ccat 0, y 0, cf 0, xt 0), history slots
        >= H' blank (he_raw 0, hemb_ok False, hcat / het / hes 0, hidx -1);
    (4) shape conventions of gather(): Mx' in {Mx, Mx + 16} (multiple of 16), Kx' a multiple of 64 in [64, max_hist], H' <= Kx', M' <= Mx'.
    Returns the list of violation strings."""
    v = []
    exp = (set(b) - REAL_ONLY) | PSEUDO_KEYS
    if set(bp) != exp:
        v.append(f"key set {sorted(set(bp) ^ exp)} differs")
    for k in set(bp) & set(b):
        if bp[k].dtype != b[k].dtype or bp[k].shape[2:] != b[k].shape[2:] or bp[k].dim() != b[k].dim():
            v.append(f"{k}: dtype / rank / trailing dims {bp[k].dtype} {tuple(bp[k].shape)} vs real {b[k].dtype} {tuple(b[k].shape)}")
    P, Mx = bp["cmask"].shape
    Kp = bp["hmask"].shape[1]
    Mx0 = b["cmask"].shape[1]
    if Mx not in (Mx0, Mx0 + 16) or Mx % 16:
        v.append(f"Mx' {Mx} not in (Mx {Mx0}, Mx + 16)")
    if Kp % 64 or not (64 <= Kp <= max_hist):
        v.append(f"Kx' {Kp} not a multiple of 64 in [64, {max_hist}]")
    src, t_all, g_all, valid = (bp[k].cpu() for k in ("pseudo_src", "pseudo_t", "pseudo_g", "pseudo_valid"))
    for p in range(P):
        i, t, g = int(src[p]), int(t_all[p]), int(g_all[p])
        tag = f"p{p} src{i} t{t} g{g}"
        Hp, Mp = int(bp["H"][p]), int(bp["M"][p])
        if Hp > Kp or Mp > Mx:
            v.append(f"{tag}: H' {Hp} > Kx' {Kp} or M' {Mp} > Mx' {Mx}")
            continue
        if not torch.equal(bp["hidx"][p, :Hp].cpu(), b["hidx"][i, t + g:t + g + Hp].cpu()):
            v.append(f"{tag}: hidx' != the source's hidx at index >= t + g")
        if (bp["hidx"][p, Hp:] != -1).any():
            v.append(f"{tag}: padded hidx' != -1")
        for k_ in ("hcat", "het", "hes"):
            if bp[k_][p, Hp:].abs().sum() > 0:
                v.append(f"{tag}: padded {k_}' not 0")
        if bp["hemb_ok"][p, Hp:].any() or bp["he_raw"][p, Hp:].abs().sum() > 0:
            v.append(f"{tag}: padded history not blank")
        if t + g + Hp != int(b["H"][i]):
            v.append(f"{tag}: the pseudo history does not run to the source's oldest stored event")
        if bool(valid[p]):
            pad = slice(Mp, Mx)
            if (bp["ce_raw"][p, pad].abs().sum() > 0 or not bool(bp["cmiss"][p, pad].all()) or bp["ccat"][p, pad].abs().sum() > 0
                    or bp["y"][p, pad].abs().sum() > 0 or bp["cf"][p, pad].abs().sum() > 0 or ("xt" in bp and bp["xt"][p, pad].abs().sum() > 0)):
                v.append(f"{tag}: padded candidate slots not blank")
    return v

"""hz2 kernels (numba). All label-free. See DECLARATION.txt for the rule each kernel implements.
Masks: dmask[row] = uint16 bitmask of TRAIN day slots (0..14) whose train snapshots / impressions count as dense for the row;
       smask[feed] = uint32 bitmask of day slots (0..23) whose sparse-source snapshots count (minus the row's dense days);
       rfac[row]   = depth factor applied to the dense part (1 for exact rows)."""
import numpy as np
import numba as nb

MSK = 3 * 3600
DAY = 86400
DAY0 = 20702
CAPF = 63 / 64


@nb.njit(parallel=True, cache=True)
def row_masks(coff, M, f4, kn, ts_q, d_q, L_q, m, dpat, dense_bits, dmask, rfac, xtest, pslot, trunc):
    """hz2 rule per candidate row. d_q = feed day slot, L_q = lag (days), m = minimum dense depth for truncated rows.
    xtest[row] = X_test (0..12); pslot[row] = slot of the listing's creation day (-30000 if capped); dmask / rfac as in the module doc.
    rfac < 0 marks nothing; truncated rows are those with code bit: stored separately by the caller through xtest and dmask."""
    for q in nb.prange(len(M)):
        d = d_q[q]
        c = d - L_q[q]
        t = ts_q[q]
        for i in range(M[q]):
            r = coff[q] + i
            x = np.float64(f4[r])
            if x < CAPF:
                u = x * 64.0
                k = int(np.floor(u))
                if k < 1:
                    k = 1
                if k > 62:
                    k = 62
                w = u - k
                a = kn[k] * (1.0 - w) + kn[k + 1] * w
                p = int(np.floor((t - 3600.0 * a + MSK) / DAY)) - DAY0
                if p < -30000:
                    p = -30000
            else:
                p = -30000
            pslot[r] = p
            xt = 0
            xa = 0
            mask = 0
            for j in range(len(dpat)):
                s = c + dpat[j]
                if s >= 0:
                    mask |= (1 << s)
                if s >= p:
                    xt += 1
                    if s >= 0:
                        xa += 1
            xtest[r] = xt
            if xa == xt:                       # EXACT replay: every test-pattern day on which the listing was alive is observed
                dmask[r] = mask
                rfac[r] = 1.0
                trunc[r] = False
            else:                              # TRUNCATED: alive before Sep 6
                trunc[r] = True
                cr = max(c, m)
                if cr > d:
                    cr = d
                cnt = 0
                mk = 0
                for s in range(0, cr):
                    if (dense_bits >> s) & 1 and cnt < xt:
                        mk |= (1 << s)
                        cnt += 1
                dmask[r] = mk
                if cnt > 0:
                    rf = xt / cnt
                    rfac[r] = rf if rf < 4.0 else 4.0
                else:
                    rfac[r] = 1.0


@nb.njit(cache=True)
def _count_in(hs_sorted, order, het, ho, c, acc, wgt):
    """acc[0..4] += wgt * occurrences of item c in a history whose items are hs_sorted (ascending) with original positions order."""
    n = len(hs_sorted)
    k = np.searchsorted(hs_sorted, c)
    while k < n and hs_sorted[k] == c:
        acc[0] += wgt
        e = het[ho + order[k]]
        if 1 <= e <= 4:
            acc[e] += wgt
        k += 1


@nb.njit(parallel=True, cache=True)
def ithist_kernel(qg, coff, M, citem, nidx, dmask, rfac, smask, DM, SM, g_sp, g_slot, g_istr, g_hoff, g_H, hitem, het, g_user,
                  lat_start, lat_list, out):
    """out[row, 0..4] = rfac * dense + sparse + own snapshot - later same-user counted snapshots (it_hist all, et1..et4)."""
    for q in nb.prange(len(qg)):
        gf = qg[q]
        m = M[q]
        co = coff[q]
        own_slot = g_slot[gf]
        own_sp = g_sp[gf]
        own_tr = g_istr[gf]
        h = g_H[gf]
        ho = g_hoff[gf]
        hs = np.empty(h, np.int32)
        for j in range(h):
            hs[j] = hitem[ho + j]
        order = np.argsort(hs)
        hss = hs[order]
        smq = np.uint32(smask[q])
        dm_any = np.uint32(0)
        acc = np.zeros(5, np.float64)
        for i in range(m):
            r = co + i
            c = citem[r]
            k = nidx[c]
            dm = np.uint32(dmask[r])
            dm_any |= dm
            sm = smq & ~dm
            rf = rfac[r]
            for e in range(5):
                acc[e] = 0.0
            if k >= 0:
                for s in range(15):
                    if (dm >> np.uint32(s)) & np.uint32(1):
                        for e in range(5):
                            acc[e] += rf * DM[k, s, e]
                for s in range(24):
                    if (sm >> np.uint32(s)) & np.uint32(1):
                        for e in range(5):
                            acc[e] += SM[k, s, e]
            counted = (own_tr and ((dm >> np.uint32(own_slot)) & np.uint32(1)) != 0) or (own_sp and ((sm >> np.uint32(own_slot)) & np.uint32(1)) != 0)
            if not counted:
                _count_in(hss, order, het, ho, c, acc, 1.0)
            for e in range(5):
                out[r, e] = acc[e]
        u = g_user[gf]
        if u != 0:
            for li in range(lat_start[gf], lat_start[gf + 1]):
                g = lat_list[li]
                gs = np.uint32(g_slot[g])
                gt = g_istr[g]
                gsp = g_sp[g]
                if not ((gt and ((dm_any >> gs) & np.uint32(1)) != 0) or (gsp and ((smq >> gs) & np.uint32(1)) != 0)):
                    continue
                hg = g_H[g]
                hog = g_hoff[g]
                gh = np.empty(hg, np.int32)
                for j in range(hg):
                    gh[j] = hitem[hog + j]
                og = np.argsort(gh)
                ghs = gh[og]
                for i in range(m):
                    r = co + i
                    dm = np.uint32(dmask[r])
                    sm = smq & ~dm
                    w = 0.0
                    if gt and ((dm >> gs) & np.uint32(1)) != 0:
                        w = rfac[r]
                    elif gsp and ((sm >> gs) & np.uint32(1)) != 0:
                        w = 1.0
                    if w > 0.0:
                        for e in range(5):
                            acc[e] = 0.0
                        _count_in(ghs, og, het, hog, citem[r], acc, w)
                        for e in range(5):
                            out[r, e] -= acc[e]


@nb.njit(parallel=True, cache=True)
def xp_kernel(coff, M, citem, nidx, dmask, rfac, IM, out):
    """out[row] = log1p(min(rfac * train impressions of the item on the dense days, 50))."""
    for q in nb.prange(len(M)):
        for i in range(M[q]):
            r = coff[q] + i
            k = nidx[citem[r]]
            n = 0.0
            if k >= 0:
                dm = np.uint32(dmask[r])
                for s in range(15):
                    if (dm >> np.uint32(s)) & np.uint32(1):
                        n += IM[k, s]
            n *= rfac[r]
            if n > 50.0:
                n = 50.0
            out[r] = np.log1p(n)


@nb.njit(parallel=True, cache=True)
def seq_kernel(qg, coff, M, citem, nidx, dmask, smask, hpos, g_hoff, g_H, hitem, g_user, g_ts, later_mask,
               p_start, dcum, p_feed, p_pos, s_start, scum, s_feed, s_pos, out):
    """feats_v2/seq_/feat_covis.py::_covis with per-row membership. out[row] = (dense invd20, dense pop, sparse invd20, sparse pop).
    Members, most recent first, at most 3000 occurrences: sparse-source postings of the days in smask & ~dmask, and train postings of the days in dmask
    whose feed is the latest of its user among those days (later_mask & dmask == 0; user-0 feeds always). Then: same non-zero user skipped;
    sparse members need ts < own ts (the stored 'constrained' rule)."""
    for q in nb.prange(len(qg)):
        gf = qg[q]
        h = g_H[gf]
        ho = g_hoff[gf]
        u = g_user[gf]
        t = g_ts[gf]
        hs = np.empty(h, np.int32)
        for j in range(h):
            hs[j] = hitem[ho + j]
        order = np.argsort(hs)
        nu = 0
        uh = np.empty(h, np.int32)
        uw = np.empty(h, np.float64)
        for jj in range(h):
            j = order[jj]
            w = 0.95 ** hpos[ho + j]
            if nu > 0 and uh[nu - 1] == hs[j]:
                if w > uw[nu - 1]:
                    uw[nu - 1] = w
            else:
                uh[nu] = hs[j]
                uw[nu] = w
                nu += 1
        buf = np.empty(3000, np.int64)
        bsp = np.empty(3000, np.bool_)
        smq = np.uint32(smask[q])
        co = coff[q]
        for i in range(M[q]):
            r = co + i
            out[r, 0] = 0.0
            out[r, 1] = 0.0
            out[r, 2] = 0.0
            out[r, 3] = 0.0
            c = citem[r]
            k = nidx[c]
            if nu == 0 or k < 0:
                continue
            dm = np.uint32(dmask[r])
            sm = smq & ~dm
            dm16 = np.uint16(dmask[r])
            nb_ = 0
            for s in range(23, -1, -1):
                if nb_ >= 3000:
                    break
                if (sm >> np.uint32(s)) & np.uint32(1):
                    lo = s_start[k] + scum[k, s]
                    e = s_start[k] + scum[k, s + 1] - 1
                    while e >= lo and nb_ < 3000:
                        buf[nb_] = e
                        bsp[nb_] = True
                        nb_ += 1
                        e -= 1
                elif s < 15 and ((dm >> np.uint32(s)) & np.uint32(1)) != 0:
                    lo = p_start[k] + dcum[k, s]
                    e = p_start[k] + dcum[k, s + 1] - 1
                    while e >= lo and nb_ < 3000:
                        if (later_mask[p_feed[e]] & dm16) == 0:
                            buf[nb_] = e
                            bsp[nb_] = False
                            nb_ += 1
                        e -= 1
            dinv = 0.0
            dpop = 0.0
            sinv = 0.0
            spop = 0.0
            for bi in range(nb_ - 1, -1, -1):
                e = buf[bi]
                if bsp[bi]:
                    f = s_feed[e]
                    p = s_pos[e]
                    if u != 0 and g_user[f] == u:
                        continue
                    if g_ts[f] >= t:
                        continue
                else:
                    f = p_feed[e]
                    p = p_pos[e]
                    if u != 0 and g_user[f] == u:
                        continue
                st = g_hoff[f]
                Ls = g_H[f]
                lo2 = max(0, p - 20)
                hi2 = min(Ls, p + 21)
                a = 0.0
                for qq in range(lo2, hi2):
                    if qq == p:
                        continue
                    x = hitem[st + qq]
                    if x == c:
                        continue
                    kk = np.searchsorted(uh[:nu], x)
                    if kk < nu and uh[kk] == x:
                        a += uw[kk] / abs(qq - p)
                if bsp[bi]:
                    spop += 1.0
                    sinv += a
                else:
                    dpop += 1.0
                    dinv += a
            out[r, 0] = dinv
            out[r, 1] = dpop
            out[r, 2] = sinv
            out[r, 3] = spop


def combine_seq(o4, rfac):
    """(invd20, pop, norm) float32 from the kernel's four parts and the depth factor."""
    rf = rfac.astype(np.float64)
    pop = np.minimum(rf * o4[:, 1] + o4[:, 3], 3000.0)
    invd = rf * o4[:, 0] + o4[:, 2]
    norm = invd / np.sqrt(pop + 1.0)
    return invd.astype(np.float32), pop.astype(np.float32), norm.astype(np.float32)

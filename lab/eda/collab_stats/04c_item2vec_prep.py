"""Prepare token sequences for item2vec: one history per (user, split) = the latest feed's history; plus
(train positive candidate -> last-10 history items) 'next-item' pairs."""
import numpy as np, polars as pl, time
from common import *
t0=time.time()
seqs=[]; 
for s in ['train','val','test']:
    f=feeds(s)
    last=f.sort('timestamp').group_by('user_id').agg(pl.col('fidx').last())['fidx'].to_numpy()
    last=np.sort(last)
    H=f['H'].to_numpy(); off=f['hist_off'].to_numpy()
    hi=np.load(f'{FLAT}/{s}/hist_item_id.npy')
    # gather
    lens=H[last]; starts=off[last]
    idx=np.repeat(starts-np.cumsum(np.r_[0,lens[:-1]]), lens)+np.arange(lens.sum())
    toks=hi[idx]
    seqs.append((toks, np.repeat(np.arange(len(last)), lens)))
    tlog(t0,f'{s}: seqs {len(last)} tokens {len(toks)}')
# concatenate with unique sequence ids
toks=np.concatenate([t for t,_ in seqs]); sid=[]; base=0
for t,s_ in seqs: sid.append(s_+base); base+=s_.max()+1
sid=np.concatenate(sid)
cnt=np.bincount(toks, minlength=NITEM)
rel=np.load(f'{OUT}/rel_items.npy')
keepm=cnt>=3; keepm[rel]=keepm[rel]|(cnt[rel]>=1)
vocab=np.nonzero(keepm)[0]
vlut=np.full(NITEM,-1,np.int32); vlut[vocab]=np.arange(len(vocab),dtype=np.int32)
vt=vlut[toks]; mk=vt>=0
print('vocab', len(vocab), 'tokens kept', mk.mean())
np.save(f'{OUT}/i2v_vocab.npy', vocab)
np.save(f'{OUT}/i2v_tokens.npy', vt[mk]); np.save(f'{OUT}/i2v_sid.npy', sid[mk].astype(np.int32))
np.save(f'{OUT}/i2v_counts.npy', cnt[vocab].astype(np.int64))
# next-item pairs from train positives
f=feeds('train'); M=f['M'].to_numpy(); H=f['H'].to_numpy()
tg=np.load(f'{FLAT}/train/cand_target.npy').astype(bool)
ci=np.load(f'{FLAT}/train/cand_item_id.npy')
pf=np.repeat(np.arange(len(f)),M)[tg]; pi=ci[tg]
hp=np.load(f'{FLAT}/train/hist_hpos.npy'); hi=np.load(f'{FLAT}/train/hist_item_id.npy'); hf=np.load(f'{FLAT}/train/hist_fidx.npy')
mh=hp<10
P=pl.DataFrame({'f':pf,'c':vlut[pi]}).filter(pl.col('c')>=0).join(pl.DataFrame({'f':hf[mh],'h':vlut[hi[mh]]}).filter(pl.col('h')>=0),on='f')
print('next-item pairs', len(P))
np.save(f'{OUT}/i2v_nextpairs.npy', np.stack([P['c'].to_numpy(),P['h'].to_numpy()]).astype(np.int32))
tlog(t0,'done')

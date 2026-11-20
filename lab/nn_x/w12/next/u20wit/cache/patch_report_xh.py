"""u20wit: one-off helper that fills the XH placeholders of reports/NEXT_1115/u20wit.md (numbers copied by hand from out/xh_read.txt)."""
import re
p = '<repo>/reports/NEXT_1115/u20wit.md'
s = open(p).read()
old4 = s[s.index("4. Seeing a feed's own inputs"):s.index("5. The witness and the CPU path")]
new4 = ("4. Whether a feed's own inputs add anything beyond the general effect of the unlabelled fold is not established: long witness minus cross-user holdout "
        "(CPU, n 8,000) = +0.00013 ± 0.00076, short minus the same = -0.00022 ± 0.00106; long witness minus the same users' unseen feeds (H0, n 5,974) = "
        "+0.00182 ± 0.00084 (@last +0.00122 ± 0.00084). B matches the teacher by the same extra amount on seen and unseen long feeds (within-feed corr +0.0064 vs +0.0065) [M]. "
        "In-sample bound C - A on the same feeds: +0.0229 ± 0.0009 short / +0.0178 ± 0.0004 long (bound only, not evidence).\n")
s = s.replace(old4, new4)
R = {
    "«XH_A»": "0.32440 (members 0.32391, 0.32488)",
    "«XH_B»": "0.32522 (members 0.32452, 0.32591)",
    "«XH_BA»": "+0.00082 ± 0.00067 (+0.00061, +0.00103); same 8,000 feeds from the GPU files +0.00075 ± 0.00066",
    "«XH_BA_POOL»": "+0.00018 ± 0.00075",
    "«XH_EXCESS_L»": "+0.00013 ± 0.00076",
    "«XH_EXCESS_S»": "-0.00022 ± 0.00106",
    "«XH_EXCESS_H»": "-0.00168 ± 0.00101",
    "«XH_SENTENCE»": ("Against the cross-user holdout replayed on the CPU (B - A +0.00082 ± 0.00067, n 8,000) the excess is +0.00013 ± 0.00076 on long and -0.00022 ± 0.00106 on short feeds; "
                      "against the same users' unseen feeds (H0) it is +0.00182 ± 0.00084 on long feeds (@last +0.00122 ± 0.00084) [M]. The two baselines differ by 0.0017 ± 0.0010: on H0 both "
                      "B - A (-0.00086 ± 0.00076) and the out-of-sample C - A (+0.00038 ± 0.00082, against +0.00143 ± 0.00040 on the cross-user holdout) read low, and base - xf0 does the same "
                      "(+0.00152 ± 0.00081 against +0.00251 ± 0.00048); I have no mechanism for it and each gap is within 2 SE. The label-free mechanism check sides with 'no feed-specific effect': "
                      "the only training signal B has on a witness feed is the teacher's distribution, and B's extra match to the teacher is the same on long feeds it trained on "
                      "(+0.00638 ± 0.00004) and on long feeds it never saw (+0.00650 ± 0.00009) [M; the conclusion is I]."),
}
for k, v in R.items():
    assert k in s, k
    s = s.replace(k, v)
old = "| H0 minus XH (same-user channel without the feed) | -0.00168 ± 0.00101 |"
assert s.count(old) == 1
s = s.replace(old, old + "\n| W0L with old p_random > 0.9 minus XH | -0.00020 ± 0.00093 |\n| W0S + W0L minus XH | +0.00003 ± 0.00075 |\n"
              "| CPU minus GPU-file value of the XH delta itself (same 8,000 feeds) | +0.00007 ± 0.00019 |")
open(p, 'w').write(s)
print('left:', sorted(set(re.findall('«[A-Z0-9_]+»', s))))

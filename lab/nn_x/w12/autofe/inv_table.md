| input | NN scaling | zeros | NaN val / train | mode mass | skew / kurt (as fed) | max abs z test | PSI trn→val | PSI val→test | between-feed var | rel. version in model | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f1 | raw | 0.000 | 0.000 / 0.000 | 0.176 | +0.3 / -1 | 1.7 | 0.026 | 0.000 | 0.32 | yes |  |
| f2 | raw | 0.154 | 0.000 / 0.000 | 0.155 | +0.1 / -1 | 1.7 | 0.003 | 0.000 | 0.31 | yes |  |
| f3 | raw | 0.000 | 0.000 / 0.000 | 0.056 | +0.2 / -1 | 1.7 | 0.030 | 0.001 | 0.17 | yes |  |
| f4 | raw | 0.000 | 0.000 / 0.000 | 0.019 | +0.0 / -1 | 1.7 | 0.014 | 0.001 | 0.05 | yes |  |
| f5 | raw | 0.013 | 0.000 / 0.000 | 0.108 | +0.1 / -1 | 1.7 | 0.031 | 0.001 | 0.07 | yes |  |
| cos_u | raw | 0.014 | 0.000 / 0.000 | 0.014 | -2.2 / 19 | 16.9 | 0.130 | 0.002 | 0.34 | yes | train→val drift |
| seq_max_all | raw | 0.000 | 0.014 / 0.012 | 0.000 | -5.9 / 41 | 8.2 | 0.118 | 0.000 | 0.83 | yes | sentinel −1.5 mixed with values; train→val drift |
| maxsim32 | raw | 0.000 | 0.014 / 0.012 | 0.000 | -5.2 / 34 | 7.8 | 0.135 | 0.000 | 0.77 | no | sentinel −1.5 mixed with values; train→val drift; mostly feed-level, no relative version |
| seq_top3 | raw | 0.000 | 0.014 / 0.012 | 0.000 | -5.9 / 41 | 8.1 | 0.114 | 0.000 | 0.83 | yes | sentinel −1.5 mixed with values; train→val drift |
| cos_hall | raw | 0.000 | 0.014 / 0.012 | 0.000 | -5.8 / 40 | 8.0 | 0.044 | 0.001 | 0.82 | no | sentinel −1.5 mixed with values; mostly feed-level, no relative version |
| cos_het1 | raw | 0.000 | 0.031 / 0.029 | 0.000 | -4.5 / 21 | 5.5 | 0.039 | 0.001 | 0.90 | no | sentinel −1.5 mixed with values; mostly feed-level, no relative version |
| cos_het2 | raw | 0.000 | 0.164 / 0.161 | 0.000 | -1.7 / 1 | 2.2 | 0.019 | 0.000 | 0.98 | no | sentinel −1.5 mixed with values; mostly feed-level, no relative version |
| cos_het3 | raw | 0.000 | 0.100 / 0.098 | 0.000 | -2.4 / 4 | 3.0 | 0.024 | 0.000 | 0.96 | no | sentinel −1.5 mixed with values; mostly feed-level, no relative version |
| cos_het4 | raw | 0.000 | 0.241 / 0.246 | 0.000 | -1.1 / -1 | 1.7 | 0.010 | 0.000 | 0.98 | no | sentinel −1.5 mixed with values; mostly feed-level, no relative version |
| cat_cnt_all | log1p | 0.158 | 0.000 / 0.000 | 0.159 | -0.2 / -1 | 1.7 | 0.010 | 0.000 | 0.24 | yes |  |
| cat_cnt_et1 | log1p | 0.228 | 0.000 / 0.000 | 0.228 | +0.0 / -1 | 1.9 | 0.020 | 0.000 | 0.27 | no |  |
| cat_cnt_et2 | log1p | 0.495 | 0.000 / 0.000 | 0.495 | +1.0 / -0 | 3.8 | 0.001 | 0.000 | 0.43 | no |  |
| cat_cnt_et3 | log1p | 0.480 | 0.000 / 0.000 | 0.481 | +0.7 / -1 | 2.3 | 0.002 | 0.000 | 0.30 | no |  |
| cat_share_all | raw | 0.146 | 0.014 / 0.012 | 0.147 | +1.2 / 0 | 2.6 | 0.005 | 0.001 | 0.33 | no |  |
| cat_min_hpos | raw | 0.287 | 0.000 / 0.000 | 0.286 | +1.8 / 1 | 2.5 | 0.019 | 0.000 | 0.17 | no |  |
| feed_cat_share | raw | 0.000 | 0.000 / 0.000 | 0.004 | +1.2 / 1 | 3.3 | 0.021 | 0.002 | 0.44 | no |  |
| item_norm | raw | 0.000 | 0.000 / 0.000 | 0.003 | -79.6 / 6384 | 171.7 | 0.000 | 0.000 | 0.01 | no | stand-in only (model uses log1p(norm)·has_emb) |
| seq_cv_invd20 | log1p | 0.887 | 0.000 / 0.000 | 0.886 | +11.7 / 189 | 27.5 | 0.011 | 0.002 | 0.03 | yes | sparse + heavy tail |
| seq_cv_pop | log1p | 0.420 | 0.000 / 0.000 | 0.420 | +0.9 / 0 | 4.4 | 0.002 | 0.012 | 0.07 | no |  |
| seq_cv_norm | log1p | 0.887 | 0.000 / 0.000 | 0.886 | +15.0 / 331 | 35.3 | 0.011 | 0.002 | 0.03 | no | sparse + heavy tail |
| it_hist_all_luo | log1p | 0.379 | 0.000 / 0.000 | 0.379 | +0.8 / -0 | 3.5 | 0.464 | 0.019 | 0.06 | yes | train→val drift |
| it_hist_et1_luo | log1p | 0.411 | 0.000 / 0.000 | 0.411 | +1.0 / 0 | 3.8 | 0.449 | 0.017 | 0.06 | no | train→val drift |
| it_hist_et2_luo | log1p | 0.689 | 0.000 / 0.000 | 0.689 | +2.0 / 4 | 6.4 | 0.201 | 0.006 | 0.04 | no | train→val drift |
| it_hist_et3_luo | log1p | 0.713 | 0.000 / 0.000 | 0.713 | +2.0 / 4 | 5.9 | 0.191 | 0.004 | 0.04 | no | train→val drift |
| it_hist_et4_luo | log1p | 0.749 | 0.000 / 0.000 | 0.748 | +2.4 / 6 | 6.3 | 0.146 | 0.003 | 0.06 | no | train→val drift |
| ctx_nd_hcent | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | -0.2 / 1 | 6.5 | 0.026 | 0.001 | 0.17 | no |  |
| ctx_nd_knn_f1 | clip±10 | 0.000 | 0.000 / 0.000 | 0.016 | +0.3 / -1 | 1.8 | 0.032 | 0.000 | 0.40 | no |  |
| ctx_nd_knn_h | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | -1.7 / 5 | 5.0 | 0.201 | 0.000 | 0.42 | no | train→val drift |
| ctx_nd_knn_h_dev | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | -1.0 / 3 | 9.0 | 0.019 | 0.000 | 0.03 | no |  |
| ctx_nd_knn_u | clip±10 | 0.000 | 0.014 / 0.026 | 0.000 | -2.1 / 8 | 4.7 | 0.173 | 0.002 | 0.48 | no | train→val drift |
| ctx_nd_knn_u_dev | clip±10 | 0.000 | 0.014 / 0.026 | 0.000 | -1.2 / 3 | 12.0 | 0.031 | 0.001 | 0.03 | no |  |
| ctx_nd_max | clip±10 | 0.000 | 0.000 / 0.000 | 0.000 | -1.2 / 2 | 9.3 | 0.107 | 0.000 | 0.18 | no | train→val drift |
| ctx_nd_top5h | clip±10 | 0.000 | 0.000 / 0.000 | 0.000 | +0.3 / -0 | 3.5 | 0.054 | 0.000 | 0.24 | no |  |
| ctx_nd_top5u | clip±10 | 0.000 | 0.014 / 0.026 | 0.000 | -0.2 / 0 | 3.2 | 0.048 | 0.002 | 0.30 | no |  |
| ctx_nd_ucent | clip±10 | 0.000 | 0.014 / 0.026 | 0.000 | +0.6 / 1 | 6.2 | 0.011 | 0.001 | 0.23 | no |  |
| ctx_hn_top1w | clip±10 | 0.875 | 0.014 / 0.012 | 0.876 | +8.5 / 97 | 18.2 | 0.008 | 0.000 | 0.08 | no | sparse + heavy tail |
| ctx_hn_rrw | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | +4.4 / 28 | 11.6 | 0.030 | 0.001 | 0.34 | no | heavy tail |
| ctx_hn_bestrk | clip±10 | 0.102 | 0.014 / 0.012 | 0.102 | +1.6 / 2 | 3.9 | 0.012 | 0.000 | 0.14 | no |  |
| ctx_hn_margin | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | -0.5 / -0 | 5.4 | 0.077 | 0.000 | 0.15 | no |  |
| ctx_hn_soft05 | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | +3.1 / 11 | 5.9 | 0.052 | 0.000 | 0.01 | no |  |
| ctx_hn_e1_rrw | clip±10 | 0.000 | 0.031 / 0.029 | 0.000 | +4.2 / 26 | 12.4 | 0.028 | 0.001 | 0.39 | no | heavy tail |
| ctx_hs_mutual | clip±10 | 0.898 | 0.014 / 0.012 | 0.898 | +9.2 / 204 | 43.7 | 0.005 | 0.000 | 0.04 | no | sparse + heavy tail |
| ctx_hs_k128 | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | +2.9 / 11 | 6.9 | 0.077 | 0.000 | 0.01 | no |  |
| ctx_hs_e1 | clip±10 | 0.000 | 0.031 / 0.029 | 0.000 | +3.1 / 11 | 6.2 | 0.055 | 0.001 | 0.01 | no |  |
| ctx_hs_t10 | clip±10 | 0.000 | 0.014 / 0.012 | 0.000 | +2.9 / 12 | 8.5 | 0.027 | 0.000 | 0.01 | no |  |
| pre2_fe_z | clip±5 | 0.000 | 0.000 / 0.284 | 0.000 | -0.7 / 0 | 6.0 | 0.425 | 0.000 | 0.00 | no | NaN→0 on 28% of train (blank regime), never on val/test; train→val drift |
| pre2_new_z | clip±5 | 0.000 | 0.000 / 0.284 | 0.000 | -0.6 / 0 | 6.0 | 0.418 | 0.000 | 0.00 | no | NaN→0 on 28% of train (blank regime), never on val/test; train→val drift |
| pre2_fe_s | clip±10 | 0.000 | 0.000 / 0.284 | 0.000 | -0.6 / 0 | 1.3 | 0.496 | 0.001 | 0.42 | yes | NaN→0 on 28% of train (blank regime), never on val/test; train→val drift |
| pre2_new_s | clip±10 | 0.000 | 0.000 / 0.284 | 0.000 | -0.5 / 0 | 1.3 | 0.475 | 0.001 | 0.44 | yes | NaN→0 on 28% of train (blank regime), never on val/test; train→val drift |
| pre2_hmax32 | clip±10 | 0.000 | 0.000 / 0.284 | 0.014 | -6.1 / 44 | 5.7 | 0.422 | 0.001 | 0.84 | yes | NaN→0 on 28% of train (blank regime), never on val/test; heavy tail; train→val drift |
| has:pre2_fe | 0/1 | 0.000 | 0.000 / 0.000 | 1.000 | +0.0 / -3 | 0.6 | 2.349 | 0.000 | 0.00 | no | train→val drift |
| M | log | 0.000 | 0.000 / 0.000 | 0.017 | -0.3 / -1 | 1.9 | 0.060 | 0.004 |  |  |  |
| H | log | 0.013 | 0.000 / 0.000 | 0.290 | -2.8 / 9 | 5.1 | 0.090 | 0.002 |  |  |  |
| user0 | raw/none | 0.954 | 0.000 / 0.000 | 0.954 | +4.4 / 17 | 4.9 | 0.000 | 0.000 |  |  |  |
| cold_share | raw/none | 0.005 | 0.000 / 0.000 | 0.029 | +0.4 / 1 | 8.6 | 3.659 | 0.360 |  |  | train→val drift |

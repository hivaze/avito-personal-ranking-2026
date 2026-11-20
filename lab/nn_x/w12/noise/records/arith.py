"""Arithmetic behind the records inventory (no data read; inputs copied from the cited logs).
inputs: lab/nn_x/w12/fixmiss/systematic/logs_s1.txt (NDCG 0.40294, redraw 0.40124 +- 0.00176 over 200 draws, per-feed-T redraw 0.39950)
        scratchpad loss_design/design/p7_ips_target.txt (random-order positive rate by display rank)
        task text (public 0.3952092, leader 0.3982556; feeds val 14423, public 8562, private 20033)"""
import math
act, rd, sd, rd_pf = 0.40294, 0.40124, 0.00176, 0.39950
print(f"actual - redraw mean = {act-rd:+.5f} = {(act-rd)/sd:+.2f} redraw sd; actual - per-feed-T redraw = {act-rd_pf:+.5f}")
for name, n in [("public", 8562), ("private", 20033)]:
    print(f"label-draw sd of one file's NDCG scaled to {name} ({n} feeds): {sd*math.sqrt(14423/n):.5f}")
print(f"gap to leader on public: {0.3982556-0.3952092:.7f}")
r = [0.0243, 0.0224, 0.0182, 0.0163, 0.0149, 0.0137, 0.0117, 0.0096, 0.0074, 0.0063]
print(f"random-order train feeds: positive rate at display rank 0 / rank 200-399 = {r[0]/r[-1]:.2f}x (hidden on val/test)")

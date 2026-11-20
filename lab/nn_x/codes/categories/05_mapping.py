"""05: candidate mapping code -> Avito category, and its tests.
Hypothesis H_rank: code = 1 + dense rank of Avito's public category id among the 50 leaf categories (code 1 unused);
codes 52/53 = two categories with ids > 116 (newer: hotels of Avito Travel, Avito Podrabotka shifts; see 04_posting_dates).
Tests (all label-free except CTR):
 T1 vertical coherence: adjusted Rand index between the hypothesised verticals and clusterings from (a) item embeddings
    (centroid / kNN), (b) history co-browsing transitions, (c) history co-occurrence; permutation null = random relabelling of
    names to codes (10,000 draws) keeping the vertical sizes.
 T2 delivery: share of (user,item) history pairs with a type 5/6/8 event (delivery-like flow) vs Avito Delivery availability.
 T3 pairwise sanity: for each code, is its nearest embedding / transition neighbour in the same vertical?
 T4 time: weekday vs weekend share of positives (jobs should be weekday-heavy), hour profile.
Output: out/mapping.parquet, out/mapping.md (table), log."""
from common_cat import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

AV = {  # code: (avito_id, name_ru, name_en, vertical, delivery_available)
    2: (9, "Автомобили", "Cars", "Transport", 0), 3: (10, "Запчасти и аксессуары", "Parts & accessories", "Transport", 1),
    4: (11, "Водный транспорт", "Water transport", "Transport", 0), 5: (14, "Мотоциклы и мототехника", "Motorcycles", "Transport", 0),
    6: (19, "Ремонт и строительство", "Repair & construction", "Home", 1), 7: (20, "Мебель и интерьер", "Furniture & interior", "Home", 1),
    8: (21, "Бытовая техника", "Household appliances", "Home", 1), 9: (23, "Комнаты", "Rooms", "RealEstate", 0),
    10: (24, "Квартиры", "Apartments", "RealEstate", 0), 11: (25, "Дома, дачи, коттеджи", "Houses & cottages", "RealEstate", 0),
    12: (26, "Земельные участки", "Land plots", "RealEstate", 0), 13: (27, "Одежда, обувь, аксессуары", "Clothing & shoes", "Personal", 1),
    14: (28, "Часы и украшения", "Watches & jewelry", "Personal", 1), 15: (29, "Детская одежда и обувь", "Kids clothing", "Personal", 1),
    16: (30, "Товары для детей и игрушки", "Kids goods & toys", "Personal", 1), 17: (31, "Настольные компьютеры", "Desktop PCs", "Electronics", 1),
    18: (32, "Аудио и видео", "Audio & video", "Electronics", 1), 19: (33, "Билеты и путешествия", "Tickets & travel", "Hobby", 0),
    20: (34, "Велосипеды", "Bicycles", "Hobby", 1), 21: (36, "Коллекционирование", "Collectibles", "Hobby", 1),
    22: (38, "Музыкальные инструменты", "Musical instruments", "Hobby", 1), 23: (39, "Спорт и отдых", "Sports & leisure", "Hobby", 1),
    24: (40, "Оборудование для бизнеса", "Business equipment", "Business", 1), 25: (42, "Коммерческая недвижимость", "Commercial property", "RealEstate", 0),
    26: (81, "Грузовики и спецтехника", "Trucks & machinery", "Transport", 0), 27: (82, "Продукты питания", "Food", "Home", 0),
    28: (83, "Книги и журналы", "Books & magazines", "Hobby", 1), 29: (84, "Телефоны", "Phones", "Electronics", 1),
    30: (85, "Гаражи и машиноместа", "Garages & parking", "RealEstate", 0), 31: (86, "Недвижимость за рубежом", "Property abroad", "RealEstate", 0),
    32: (87, "Посуда и товары для кухни", "Tableware & kitchen", "Home", 1), 33: (88, "Красота и здоровье", "Beauty & health", "Personal", 1),
    34: (89, "Собаки", "Dogs", "Animals", 0), 35: (90, "Кошки", "Cats", "Animals", 0), 36: (91, "Птицы", "Birds", "Animals", 0),
    37: (92, "Аквариум", "Aquarium", "Animals", 0), 38: (93, "Другие животные", "Other animals", "Animals", 0),
    39: (94, "Товары для животных", "Pet supplies", "Animals", 1), 40: (96, "Планшеты и электронные книги", "Tablets & e-readers", "Electronics", 1),
    41: (97, "Игры, приставки и программы", "Games & consoles", "Electronics", 1), 42: (98, "Ноутбуки", "Laptops", "Electronics", 1),
    43: (99, "Оргтехника и расходники", "Office equipment", "Electronics", 1), 44: (101, "Товары для компьютера", "PC components & accessories", "Electronics", 1),
    45: (102, "Охота и рыбалка", "Hunting & fishing", "Hobby", 1), 46: (105, "Фототехника", "Photo equipment", "Electronics", 1),
    47: (106, "Растения", "Plants", "Home", 0), 48: (111, "Вакансии", "Vacancies", "Jobs", 0), 49: (112, "Резюме", "CVs / resumes", "Jobs", 0),
    50: (114, "Предложения услуг", "Services", "Services", 0), 51: (116, "Готовый бизнес", "Ready business", "Business", 0),
    52: (None, "Отели (Авито Путешествия)", "Hotels (Avito Travel)", "Travel", 0), 53: (None, "Подработка (Авито Подработка)", "Shift gigs (Avito Podrabotka)", "Jobs", 0),
}
codes = np.array(sorted(AV))
vert = np.array([AV[c][3] for c in codes])
# Avito top-level of the public tree: Transport, RealEstate, Jobs, Services, Personal, Home, Electronics, Hobby, Animals, Business (+ Travel)
cs = pl.read_parquet(OUT / "cand_stats.parquet"); hs = pl.read_parquet(OUT / "hist_stats.parquet").with_columns(pl.col("code").cast(pl.Int64))
em = np.load(OUT / "emb_mats.npz"); hm = np.load(OUT / "hist_mats.npz")


def cluster(sim, k):
    D = sim.max() - sim; np.fill_diagonal(D, 0); D = (D + D.T) / 2
    return fcluster(linkage(squareform(D, checks=False), "average"), k, "maxclust")


def lift(T):
    T = T[np.ix_(codes, codes)].astype(float).copy()
    np.fill_diagonal(T, 0)
    r = T.sum(1, keepdims=True); c = T.sum(0, keepdims=True)
    L = T * T.sum() / np.maximum(r * c, 1e-9)
    L = (L + L.T) / 2
    return np.log(np.maximum(L, 1e-3))


def cooc_lift(C):
    C = C[np.ix_(codes, codes)].astype(float).copy()
    n = np.diag(C).copy(); N = n.max() * 1.0  # approx number of histories ~ reach of the most-reached category; use PMI on Jaccard instead
    J = C / np.maximum(n[:, None] + n[None, :] - C, 1)
    np.fill_diagonal(J, 0)
    # normalise by expected under independence (row/col sums)
    E = np.outer(J.sum(1), J.sum(0)) / J.sum()
    return np.log(np.maximum(J / np.maximum(E, 1e-12), 1e-3))


sims = {
    "emb_centroid": em["centroid_cos"][np.ix_(codes, codes)],
    "emb_knn": np.log1p(((lambda K: (K / np.maximum(K.sum(1, keepdims=True), 1)) / np.maximum(em["pool_n"][None, :] / em["pool_n"].sum(), 1e-9))(em["knn_conf"]))[np.ix_(codes, codes)]),
    "trans_all_TR": lift(hm["trans_all_TR"]), "trans_t1_TR": lift(hm["trans_t1_TR"]), "trans_all_VT": lift(hm["trans_all_VT"]),
    "cooc_TR": cooc_lift(hm["cooc_TR"]),
}
sims["emb_knn"] = (sims["emb_knn"] + sims["emb_knn"].T) / 2
yv = pl.Series(vert).cast(pl.Categorical).to_physical().to_numpy()
rng = np.random.default_rng(0)
K = len(np.unique(vert))
log("hypothesised verticals:", {v: [int(c) for c in codes[vert == v]] for v in np.unique(vert)})
res_ari = {}
for nm, S in sims.items():
    for k in [K, 12]:
        lab = cluster(S, k)
        ari = adjusted_rand_score(yv, lab)
        null = np.array([adjusted_rand_score(rng.permutation(yv), lab) for _ in range(10000)])
        res_ari[(nm, k)] = (ari, (null >= ari).mean(), null.mean(), null.std())
        groups = {}
        for c, l in zip(codes, lab):
            groups.setdefault(l, []).append(int(c))
        log(f"T1 {nm:13s} k={k}: ARI {ari:.3f} (null mean {null.mean():.3f} sd {null.std():.3f}, p={max((null >= ari).mean(), 1e-4):.4f}); clusters {sorted(groups.values(), key=lambda g: g[0])}")
# T1b nearest-neighbour vertical agreement (no clustering): share of codes whose top-1 / top-3 neighbours are in the same vertical
for nm, S in sims.items():
    S2 = S.copy(); np.fill_diagonal(S2, -np.inf)
    top = np.argsort(-S2, 1)[:, :3]
    same1 = (vert[top[:, 0]] == vert).mean(); same3 = (vert[top] == vert[:, None]).mean()
    # null: expected same-vertical share for a random neighbour
    sizes = pl.Series(vert).value_counts()["count"].to_numpy()
    exp = ((sizes * (sizes - 1)).sum()) / (len(vert) * (len(vert) - 1))
    log(f"T1b {nm:13s}: top-1 neighbour in same vertical {same1:.3f}, top-3 {same3:.3f} (random {exp:.3f})")
# T2 delivery-like events
pm = hs.filter(pl.col("code").is_in(codes.tolist())).sort("code")
msg = pm["p_msg_pair_VT"].to_numpy(); deliv = np.array([AV[c][4] for c in codes])
from sklearn.metrics import roc_auc_score
log("T2 delivery: AUC of p(5/6/8 pair) for Avito-Delivery-available categories", roc_auc_score(deliv, msg))
log("T2 delivery: max p(5/6/8) among non-delivery codes", {int(c): round(float(m), 4) for c, m, d in zip(codes, msg, deliv) if d == 0 and m > 0.005},
    "min among delivery codes", {int(c): round(float(m), 4) for c, m, d in zip(codes, msg, deliv) if d == 1 and m < 0.08})
# T4 weekday / weekend: positive share by day-of-week relative to the category's average (train, all feeds)
z = np.load(OUT / "dow7.npz"); P7, N7 = z["pos"], z["n"]
ctr7 = P7 / np.maximum(N7, 1); base7 = P7.sum(0) / N7.sum(0)
rel = ctr7 / base7[None, :]
wk_rel = (rel[:, 5:].mean(1) / rel[:, :5].mean(1))
z = np.load(OUT / "hour24.npz"); P24, N24 = z["pos"], z["n"]
ctr24 = P24 / np.maximum(N24, 1); base24 = P24.sum(0) / N24.sum(0); rel24 = ctr24 / base24[None, :]
night = rel24[:, [0, 1, 2, 3, 4, 5]].mean(1) / rel24.mean(1); work = rel24[:, 9:18].mean(1) / rel24.mean(1); eve = rel24[:, 19:24].mean(1) / rel24.mean(1)
# relative exposure: positive-mix share of the category by weekday vs weekend
psh = P7 / P7.sum(0, keepdims=True)
wk_mix = psh[:, 5:].mean(1) / np.maximum(psh[:, :5].mean(1), 1e-12)
order = np.argsort(wk_rel[codes])
log("T4 weekend/weekday relative CTR (train all feeds), lowest 8:", [(int(codes[i]), AV[codes[i]][2], round(float(wk_rel[codes[i]]), 3)) for i in order[:8]])
log("T4 highest 8:", [(int(codes[i]), AV[codes[i]][2], round(float(wk_rel[codes[i]]), 3)) for i in order[::-1][:8]])
log("T4 weekend/weekday positive-mix share, lowest 6:", [(int(c), AV[c][2], round(float(wk_mix[c]), 3)) for c in codes[np.argsort(wk_mix[codes])][:6]],
    "highest 6:", [(int(c), AV[c][2], round(float(wk_mix[c]), 3)) for c in codes[np.argsort(-wk_mix[codes])][:6]])
log("T4 working-hours (9-17 MSK) relative CTR, top 6:", [(int(c), AV[c][2], round(float(work[c]), 3)) for c in codes[np.argsort(-work[codes])][:6]],
    "bottom 6:", [(int(c), AV[c][2], round(float(work[c]), 3)) for c in codes[np.argsort(work[codes])][:6]])
# per-code nearest neighbours (embedding centroid, kNN, transitions) with names
rows = []
cs_d = {r["code"]: r for r in cs.iter_rows(named=True)}; hs_d = {r["code"]: r for r in hs.iter_rows(named=True)}
for i, c in enumerate(codes):
    def nb(S):
        s = S[i].copy(); s[i] = -np.inf
        return [int(codes[j]) for j in np.argsort(-s)[:3]]
    r = cs_d[c]; h = hs_d[c]
    rows.append(dict(code=int(c), avito_id=AV[c][0], name_ru=AV[c][1], name_en=AV[c][2], vertical=AV[c][3], delivery=AV[c][4],
                     share_val=r["share_val"], share_trP=r["share_trP"], ctr_val=r["ctr_val"], ctr_trR=r["ctr_trR"], ctr_trP=r["ctr_trP"],
                     items_train=r["items_train"], rows_per_item=r["rows_per_item_train"], age_med_d=r["age_med_h"] / 24, f2_at_mode=r["f2_at_mode"],
                     f2_below_mode=r["f2_below_mode"], p_deliv_pair=h["p_msg_pair_VT"], p_contact_pair=h["p_ct_pair_VT"], fav_share=h["et4_VT"],
                     et3_over_et2=h["et3_VT"] / max(h["et2_VT"], 1e-9), knn_purity=float(em["knn_conf"][c, c] / max(em["knn_conf"][c].sum(), 1)),
                     nb_emb=str(nb(sims["emb_centroid"])), nb_knn=str(nb(sims["emb_knn"])), nb_trans=str(nb(sims["trans_all_TR"])), nb_cooc=str(nb(sims["cooc_TR"])),
                     wkend_rel_ctr=float(wk_rel[c]), work_hours_rel_ctr=float(work[c])))
df = pl.DataFrame(rows)
df.write_parquet(OUT / "mapping.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(400); pl.Config.set_fmt_str_lengths(40)
print(df.select(["code", "name_en", "vertical", "nb_emb", "nb_knn", "nb_trans", "nb_cooc"]))
# history transition neighbours: same-vertical rate
for nm in ["trans_all_TR", "cooc_TR", "emb_centroid", "emb_knn"]:
    col = {"trans_all_TR": "nb_trans", "cooc_TR": "nb_cooc", "emb_centroid": "nb_emb", "emb_knn": "nb_knn"}[nm]
    same = np.mean([AV[int(eval(s)[0])][3] == AV[c][3] for c, s in zip(df["code"], df[col])])
    log(f"T3 {nm}: top-1 neighbour same vertical {same:.3f}")
log("done")

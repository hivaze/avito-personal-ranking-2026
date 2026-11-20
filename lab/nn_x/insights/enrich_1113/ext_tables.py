"""External-knowledge tables per Avito leaf code (codes per reports/CODES_DECODING_1113.md section 3.1).

These values encode knowledge from OUTSIDE the dataset (market seasonality, listing lifetimes, purchase cycles).
They are label-free constants. The S1 late-September index is finalised from the public sources listed in
web/SOURCES.txt before S1 is scored (see PREDECLARED.txt).
"""
import numpy as np

NAMES = {2: "Cars", 3: "Parts & accessories", 4: "Water transport", 5: "Motorcycles", 6: "Repair & construction",
         7: "Furniture & interior", 8: "Household appliances", 9: "Rooms", 10: "Apartments", 11: "Houses / dachas",
         12: "Land plots", 13: "Clothing & shoes", 14: "Watches & jewellery", 15: "Kids clothing", 16: "Kids goods & toys",
         17: "Desktop PCs", 18: "Audio & video", 19: "Tickets & travel", 20: "Bicycles", 21: "Collectibles",
         22: "Musical instruments", 23: "Sports & leisure", 24: "Business equipment", 25: "Commercial property",
         26: "Trucks & machinery", 27: "Food", 28: "Books & magazines", 29: "Phones", 30: "Garages & parking",
         31: "Property abroad", 32: "Tableware & kitchen", 33: "Beauty & health", 34: "Dogs", 35: "Cats", 36: "Birds",
         37: "Aquarium", 38: "Other animals", 39: "Pet supplies", 40: "Tablets & e-readers", 41: "Games & consoles",
         42: "Laptops", 43: "Office equipment", 44: "PC components", 45: "Hunting & fishing", 46: "Photo equipment",
         47: "Plants", 48: "Vacancies", 49: "CVs", 50: "Services", 51: "Ready business", 52: "Hotels", 53: "Shift gigs"}

# ---- S1: late-September (Sep 22-29) demand index vs early September, per leaf. {-2..+2}. Finalised after the web scan.
# Rationale per entry is in web/SOURCES.txt (Russian retail seasonality: winter-tyre season starts late Sep / Oct,
# dacha / garden / water-sport / bicycle season ends, autumn clothing and kids' school-season goods peak in Sep,
# heating / repair before the Oct 1 heating season, rental demand peaks Aug-Sep (students) then cools, hunting season opens).
S1_SEASON = {2: 0, 3: +2, 4: -1, 5: -2, 6: +1, 7: +1, 8: +1, 9: -1, 10: 0, 11: -1, 12: -1, 13: +1, 14: 0, 15: 0, 16: -1,
             17: 0, 18: 0, 19: 0, 20: -2, 21: 0, 22: +1, 23: 0, 24: +1, 25: 0, 26: +1, 27: 0, 28: 0, 29: +1, 30: 0,
             31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: +1, 43: +1, 44: 0,
             45: +1, 46: 0, 47: 0, 48: +1, 49: +1, 50: +1, 51: +1, 52: 0, 53: +1}  # FINAL (web/SOURCES.txt, 19:3x UTC)

# ---- C1: purchase-cycle class (how long a buyer keeps considering / re-viewing before acting). long / medium / short.
CYCLE = {2: "long", 3: "medium", 4: "long", 5: "long", 6: "medium", 7: "medium", 8: "medium", 9: "long", 10: "long",
         11: "long", 12: "long", 13: "short", 14: "medium", 15: "short", 16: "short", 17: "medium", 18: "medium",
         19: "short", 20: "medium", 21: "medium", 22: "medium", 23: "medium", 24: "long", 25: "long", 26: "long",
         27: "short", 28: "short", 29: "medium", 30: "long", 31: "long", 32: "short", 33: "short", 34: "medium",
         35: "medium", 36: "short", 37: "short", 38: "short", 39: "short", 40: "medium", 41: "medium", 42: "medium",
         43: "medium", 44: "medium", 45: "medium", 46: "medium", 47: "short", 48: "long", 49: "long", 50: "medium",
         51: "long", 52: "short", 53: "short"}

# ---- C3: typical listing lifetime on the platform, days (public market facts / common knowledge of exposure times:
# realty sale ~45-60 d, rental ~14 d, cars ~30 d, parts ~60 d (dealer stock), jobs ~30 d, services ~90 d (evergreen),
# clothing ~30 d, electronics ~20 d, animals ~30 d, plants ~20 d, food ~15 d, tickets ~10 d, hotels ~365 d).
LIFE_D = {2: 30, 3: 60, 4: 45, 5: 30, 6: 60, 7: 30, 8: 25, 9: 14, 10: 45, 11: 60, 12: 90, 13: 30, 14: 45, 15: 30,
          16: 30, 17: 25, 18: 25, 19: 10, 20: 25, 21: 90, 22: 45, 23: 30, 24: 60, 25: 90, 26: 45, 27: 15, 28: 60,
          29: 20, 30: 60, 31: 120, 32: 45, 33: 30, 34: 30, 35: 30, 36: 30, 37: 30, 38: 30, 39: 45, 40: 20, 41: 20,
          42: 20, 43: 45, 44: 25, 45: 45, 46: 30, 47: 20, 48: 30, 49: 30, 50: 90, 51: 90, 52: 365, 53: 14}


def table(d, default=0.0, n=64):
    t = np.full(n, default, np.float64)
    for k, v in d.items():
        t[k] = v
    return t


# ---- C4: directional category complements (history category -> candidate category), external product-logic knowledge.
COMPLEMENT = {
    2: [3, 30], 3: [2, 5, 26], 5: [3], 26: [3], 4: [45], 45: [4, 23],
    10: [7, 8, 6, 32], 11: [6, 47, 7], 12: [6, 11, 47], 9: [7], 30: [2, 3], 25: [24, 51], 51: [25, 24], 24: [25],
    15: [16], 16: [15], 13: [14, 33], 14: [13],
    34: [39], 35: [39], 36: [39], 37: [39], 38: [39], 39: [34, 35],
    17: [44], 44: [17, 42], 42: [44, 43], 29: [40, 18], 41: [18, 17], 46: [18],
    48: [49, 53], 49: [48], 53: [48],
    20: [23], 23: [20, 45], 22: [18], 28: [21], 19: [52], 52: [19],
    6: [7, 8], 7: [32, 6], 8: [32],
}


def complement_matrix(n=64):
    """m[c_hist, c_cand] = 1 if a history event in c_hist is a complement cue for a candidate in c_cand."""
    m = np.zeros((n, n), np.float64)
    for src, dsts in COMPLEMENT.items():
        for d in dsts:
            m[src, d] = 1.0
    return m

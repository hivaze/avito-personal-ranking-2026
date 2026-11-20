"""s11: write the source mapping table (inference + confidence + key evidence) -> out/s11_source_mapping.csv.
Numbers are from out/s01..s09 logs (val unless noted)."""
import polars as pl
rows = [
 ("s0", "unknown/missing surface", "F6 unknown", "Surface missing for a type-1/4/7 action (deep link / push / external / item page reached without a tracked surface); also the only value for types 2,3,5,6,8,9,10", "high (missing), low (which)",
  "README: 0 = unknown/missing; types 2,3,5,6,8,9,10 are 100% s0; views s0 = 4.3%; contact_before on views 7.2% (s1 1.2%); type-4/s0 has older contact 28%, older type-4 44%; A->B: new type-1/s0 events were A candidates 3.6% (28% of them A positives)"),
 ("s1", "main-page recommendation feed", "F1 recommendation feed (target surface)", "The surface on which the competition feeds are shown; type-1/4/7 events started from it", "high",
  "A->B pairs (60K users, 20.5K pairs <=1d): oldest B event of A's positives = (1,s1) 83%, (4,s1) 2.7%, (7,s1) 0.6%, other sources <0.1%; new (1,s1) events in B were A candidates 13.2% (s2 0.45%, s4 0.9%) and 94% of those were A positives; val history items that were candidates in the user's train feeds 1.1% (80% of them positives); 33.8% of all events, 64.5% of views, in 97-99% of feeds; lowest consecutive-view similarity (cos 0.64, same-cat 50%); items most often in the candidate pool (9.1% val pool, 63.5% train pool) and youngest (median age 865 h)"),
 ("s2", "search / catalogue listing (SERP)", "F3 search", "Views and snippet actions from search results or category listings", "high (search-like), medium (search vs category browse)",
  "13.8% of events, 23.7% of views; s2->s2 consecutive views cos 0.787, same-cat 70% (query result lists), s1->s2 only 0.58; contact follow-up 7.2% vs 4.1% for s1; A->B: new (1,s2) events are A candidates 0.45%, P(pos|first event (1,s2)) 0.016; items older (median 1403 h) and less in the candidate pool (4.5%); anonymous share 0.68x logged"),
 ("s3", "favourites list (saved items)", "F4 saved-items / re-engagement", "Actions on items from the user's saved-items list; mostly type 4 (61%) and type 7 (34%), few views (5%)", "medium-high",
  "type mix 5/61/34 (1/4/7); type-4/s3 events: 39% have an older view of the item (s1 12%), 40% an older type 4; nearest older view is s1 (49%) or s2 (36%), never s3; views from s3: 50% re-views (s1 1%); cross-source view path 1->3 is the most common (704 items); A->B: s3 events on A candidates are A positives 83-97%; no-embedding 25% (closed listings kept in the list), mean hpos 135; anonymous O/E 0.27 (account-bound); s3<->s10 adjacent-transition lift 3.2-3.5"),
 ("s4", "item-card 'similar listings' block (item-to-item recs)", "F2 other recommendation surfaces", "Views of items recommended on another item's card", "medium-high",
  "cos with the immediately older view 0.76 (median 0.83, >0.9 36%), same-cat 74%, whatever the previous view's source (s1->s4 0.70, s2->s4 0.74, s0->s4 0.74; s4->s4 0.86 = hopping); feed s1->s1 0.65; 1.8% of events, 3.2% of views, in 57% of feeds; anonymous over-represented (1.8x share in user-0 feeds: SEO landings on cards); type-10 O/E 1.05; A-cand 0.9%, A-pos 3.5%"),
 ("s5", "goods-only, login-only search-like surface", "F3 search-like (goods)", "Candidates: saved-search results / seller store / delivery-goods listing; not identifiable", "low",
  "91% of views in the 36 categories that have types 5/6/8 (s1 68%); anonymous O/E 0.09 (login-bound); s5->s5 consecutive cos 0.79 (search-like), burst lift 17; contact follow-up 6.6%, type-10 follow-up 0.07% (others ~1%); contact_before 3.4% (views) / 16% (type 4) / 21% (type 7); candidate-pool share 3.0% (lowest); A-cand 2.2% (15% pos)"),
 ("s6", "secondary personalised recommendation feed", "F2 other recommendation surfaces", "A scrollable recommendation surface fed by the same candidate generator but not the logged feed (e.g. another recs screen / vertical home / platform variant)", "medium (recs), low (which)",
  "category mix nearest to s1 (JS 0.017); candidate-pool share ~s1 (8.1% val, 13.2% test); B's new s6 views were A candidates 1.8% (4x s2) but only 1.9% of those were A positives (not the target surface); very bursty (prev view same source 47%, lift 39-42); consecutive cos 0.68; anonymous O/E 0.35, type-10 O/E 0.62; night share 0.87x"),
 ("s7", "vertical surface of the no-delivery cluster 10/11/12 (real-estate-like: map / developer or complex page)", "F5 vertical-specific", "Browsing near-duplicate listings within categories 10/11/12", "medium (vertical), low (which)",
  "categories 10/11/12 = 57/21/7% (JS vs s1 0.68); 98.7% of views in categories without types 5/6/8; consecutive cos median 0.89 (s7->s7 0.92); type 4 = 42% of its events and 9% favourite follow-up; items old (median 2178 h) and rarely in the pool (3.3%); A-cand 0.5%, 0 positives"),
 ("s8", "unidentified surface (old/closed items, cats 50/13/3)", "unknown", "", "low",
  "0.1% of events; types 55/27/18; no-embedding 23%; items oldest (median 2800 h); consecutive cos 0.64; adjacent transitions to s9 / s11 lift 2.7; work-hours-leaning"),
 ("s9", "vertical recommendations for no-delivery verticals (logged-out heavy)", "F2/F5", "", "low",
  "cats 48/2/10 (78% no-delivery categories); anonymous O/E 1.12 and 2.1x in user-0 feeds; type-10 O/E 1.31; pool share ~s1 (7-13%); burst lift ~300; A-cand 1.7% (17% pos)"),
 ("s10", "saved-items-adjacent list (re-engagement)", "F4 saved-items / re-engagement", "", "low",
  "adjacent-transition lift with s3 3.2-3.5 both ways; types 46/33/21; views 9% re-views; no-embedding 27%; train-pool share lowest (38%); A->B: 62% of A-candidate items acted on via s10 were A positives"),
 ("s11", "goods-only surface concentrated in categories 13/29", "F3 search-like (goods)", "", "low",
  "100% of views in delivery categories; cats 13 44%, 29 29% (JS vs s1 0.46); consecutive cos 0.71 (>0.9 32%); messaging O/E 1.18"),
 ("s12", "login-only surface, mix like s4/s2", "unknown", "", "low",
  "never in user-0 feeds; category JS nearest s4 (0.033); type-10 follow-up 0; transitions s4<->s12 lift 1.7-1.8"),
 ("s13", "favourites-like list (vertical / alternate)", "F4 saved-items / re-engagement", "", "low",
  "type mix 9/65/26 like s3; never anonymous; no-embedding 17-25%; old (hpos 148); cats 10/11/50; cluster with s14/s16 in transitions"),
 ("s14", "vertical surface in category 10 (logged-out heavy)", "F5 vertical-specific", "", "low",
  "cat 10 50-61%; anonymous O/E 1.57, type-10 O/E 1.68; consecutive cos median 0.86-0.91; transitions with s13/s16"),
 ("s15-s30", "rare tail", "unknown", "<0.006% of events each; s17 = 96% type 4 in cats 10/11; s21 = cat 48 only; s22 = cats 29/40; s30 appears once in train", "none", "too rare to identify"),
]
df = pl.DataFrame(rows, schema=["code", "label", "family", "what_it_is", "confidence", "key_evidence"], orient="row")
df.write_csv("<repo>/nn_x/codes/sources/out/s11_source_mapping.csv")
print(df.select("code", "label", "family", "confidence"))

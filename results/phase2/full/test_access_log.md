# Phase 2 official-test access log

Freeze manifest: `250f974af64b67340eea34f4af174c4227561a6242d37c1335357793777a7662` (committed 2026-11-17, before any Phase 2 test access).
Run by Neelanshu on 2026-11-19 IST after the team agreed the freeze. Each attempt ran the unchanged frozen rows.

| # | Approx. UTC | Outcome | Test data read? |
|---:|---|---|---|
| 1 | 2026-11-18, before 18:50 | Official IDX download failed (local Python SSL certificate error) before any file reached disk | No |
| 2 | 2026-11-18 18:50:13 | `selected_cnn` scored (acc 0.9214, F1 0.9207), then `final_test` crashed on the first LR head: it called `.predict` on the joblib dict bundle | Yes (CNN row printed) |
| 3 | 2026-11-18 18:50:34 | Same frozen rows, after a one-line fix (`bundle["pipeline"].predict`, matching `fusion.py`). All four rows written | Yes |

The fix changes how the code loads a frozen artifact. It changes no model, hash, row or selection rule.
`verify_freeze` passed on every attempt. Attempt 3 reproduced attempt 2's CNN numbers exactly.
`test_access_receipt.json` holds the timestamp of the last opening only, so this file records all three.

"""Phase 2: small CNN training runner, controlled ablations and HOG/embedding fusion.

Lives in a subpackage so the Phase 1 source-file hashes recorded in
results/*/protocol.json (src/clothing_ml/*.py, non-recursive) stay unchanged.

Interfaces (team handoff, Phase 2 guide p.9):
- data.py   image float32 [B,1,28,28], label int64 [B], original ID int64 [B]
            (stand-in until Sudeshhna's adapter lands; same contract)
- models.py logits [B,10]; .embed() -> [B,64] after dense ReLU, before dropout
            (stand-in until Puneet's designs land; same contract)
- runner.py training runner, best-checkpoint rule, run record (Samarth)
- select.py validation-only CNN selection, candidate table, freeze, test (Samarth)
- fusion.py optional HOG + CNN-embedding fusion with LR controls (Samarth)
"""

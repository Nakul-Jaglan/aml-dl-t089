"""Phase 2: small CNN training runner, controlled ablations and HOG/embedding fusion.

Lives in a subpackage so the Phase 1 source-file hashes recorded in
results/*/protocol.json (src/clothing_ml/*.py, non-recursive) stay unchanged.

Interfaces (team handoff, Phase 2 guide p.9):
- data.py   image float32 [B,1,28,28], label int64 [B], original ID int64 [B]
            (ClothingImages, make_loaders, check_batch and fit-only translation policies)
- models.py SmallCNN architecture, parameter breakdown and 64-value embed/encode interface (Puneet)
- runner.py training runner, best-checkpoint rule, run record (Samarth)
- select.py validation-only CNN selection, candidate table, freeze, test (Samarth)
- fusion.py optional HOG + CNN-embedding fusion with LR controls (Samarth)
"""
from .models import (
    SmallCNN,
    build_model,
    count_parameters,
    get_parameter_breakdown,
    parameter_count,
    shape_table,
)

__all__ = [
    "SmallCNN",
    "build_model",
    "count_parameters",
    "get_parameter_breakdown",
    "parameter_count",
    "shape_table",
]

# Completed Phase 2 integration

Nakul's integration brings together the data adapter and regularization audit,
CNN architecture, training and hybrid experiments, and final evaluation. The
master notebook, IEEE/LaTeX report and presentation use the reviewed run files.

## Branch review on 19 November 2026

| Branch | Review and integration |
|---|---|
| `phase1/classical-ml` | Already in main through the Phase 1 PRs. |
| `phase1/data-quality-sudeshhna` | Already in main through PR #2. |
| `phase1/hinglish-presenter-notes` | Already in main through PR #4. |
| `phase2/training-samarth` | Already in main through PR #5. |
| `phase2/cnn-architecture-puneet` | Architecture merged in PR #6. A later main-synchronization merge added no distinct file changes and is included in the final integration ancestry. |
| `phase2/evaluation-neelanshu` | Fork branch merged in PR #7. |
| `phase2/data-adapter-sudeshhna` | Reviewed and merged through PR #8. Contribution-log and package-export conflicts were resolved by preserving both components. |

PR #8 also installs CPU PyTorch in CI so the DL checks run with the original
integrity suite. Local verification passed 60 tests. The selected CNN reloaded
and matched all 12,000 saved validation predictions. Existing original-test
predictions, selection manifests and freeze hashes remain the measured source.

## Final deliverables

- `notebooks/01_phase2_master.ipynb` and its HTML companion.
- `report/phase2/main.tex` and `main.pdf`, with measured tables, portable figures
  and cited research. The private `submission.pdf` uses the supplied author IDs.
- `presentation/phase2/Phase_2_Clothing_Category_Recognition.pptx`, with short
  Hinglish presenter notes, editable tables and charts.
- `results/phase2/integration/`, the expanded metric audit and reporting tables.
- `docs/phase2_literature_review.md`, linking the design to primary research.

## Protocol retained for the final comparison

All final models use the same 48,000 fit, 12,000 validation and 10,000 official
test IDs. Four CNN configurations at seed 42 form the selection set. Checkpoints
maximize validation macro F1. The selected CNN and three LR-head controls enter
the preserved freeze. The later probability-0.5 shift ablation is validation-only
and sits outside that choice. The trained third convolution has 128 channels
and the selected CNN has 167,114 trainable values.

The encoder is supervised by fit labels and selected on validation. LR-head
selection subsequently reuses the same validation membership. Validation values
therefore serve development comparison. Official test was previously examined
in Phase 1, so the report describes a comparison on that public benchmark.
The access log records the fixed-row retry after an artifact-loading repair.

## Reviewed outcome

| Pipeline | Test accuracy | Macro F1 | Shirt recall | Recorded CPU ms/image |
|---|---:|---:|---:|---:|
| HOG + SVM | 90.91% | 0.9090 | 73.3% | 19.748 |
| Selected CNN-3 | 92.14% | 0.9207 | 71.7% | 0.312 |
| Embedding + LR control | 92.21% | 0.9218 | 74.6% | 0.315 |
| HOG + LR control | 87.60% | 0.8757 | 66.1% | 0.205 |
| Fusion + LR | 92.06% | 0.9206 | 76.0% | 0.518 |

The CNN gains 1.23 percentage points accuracy against SVM. The paired bootstrap
95% interval is +0.73 to +1.75 points over aligned test-ID resamples. Shirt recall
falls 1.6 points while Shirt precision and F1 rise. Fusion falls below the
embedding-only control on validation and official-test macro F1. The highest
aggregate control result remains clearly identified as a control row.

## Reproduce the reporting work

Use the repo's Phase 2 Python environment from the root directory.

```bash
python -m pip install -e '.[dev,phase2]'
python -m pytest -q
python -m clothing_ml.phase2.reporting
python scripts/build_phase2_report.py
python scripts/build_phase2_notebook.py
python -m jupyter nbconvert --execute --to notebook --inplace notebooks/01_phase2_master.ipynb
python -m jupyter nbconvert --to html notebooks/01_phase2_master.ipynb
work/tectonic --keep-logs --outdir report/phase2 report/phase2/main.tex
python scripts/prepare_phase2_slides.py
```

The Tectonic binary is a local ignored build dependency. An installed Tectonic
or an IEEEtran-compatible LaTeX compiler can compile the portable report folder.
The deck's JS source uses Artifact Tool from the bundled Node runtime. The
editable PowerPoint is available directly for normal slide editing.

`phase2.reporting` checks 31 full-profile prediction files, original ID membership,
authoritative labels, independent confusion-count metrics and checkpoint hashes.
It writes only its integration tables and figures. It reads original completed
runs and preserves their selection history. The recorded one-machine timing
is reused with its metadata.

## Nakul's rehearsal

Explain the protocol, one positive aggregate finding, one class trade-off and
one unsupported hybrid hypothesis. Open test ID 40 in the notebook and trace its
true Shirt label and paired predictions into the confusion count. Be ready to
explain why a public benchmark already seen in Phase 1 and a single training
seed limit the inference.

"""Prepare measured IEEE report tables, macros and portable figure assets."""
from pathlib import Path
import shutil

import pandas as pd

from clothing_ml.phase2.reporting import CNN_ID, SVM_ID, reviewed_evidence
from clothing_ml.utils import ROOT, read_json

REPORT = ROOT / "report/phase2"
GENERATED = REPORT / "generated"
FIGURES = REPORT / "figures"
for folder in [GENERATED, FIGURES]:
    folder.mkdir(parents=True, exist_ok=True)
evidence = reviewed_evidence()
comparison = evidence["comparison"]
by_row = comparison.set_index("row")
cnn, svm = by_row.loc["selected_cnn"], by_row.loc["hog_svm"]
uncertainty = read_json(ROOT / "results/phase2/full/evaluation/cnn_vs_svm_deltas.json")["paired_bootstrap"]


def finish(rows):
    return "\n".join(rows) + "\n"


rows = [f"{r.model} & {r.val_macro_f1:.4f} & {100*r.test_accuracy:.2f} & {r.test_macro_f1:.4f} & {100*r.shirt_recall:.1f} \\\\" for r in comparison.itertuples()]
(GENERATED / "comparison.tex").write_text(finish(rows))
rows = [f"{r.name.replace('_', ' ')} & {r.val_macro_f1:.4f} & {100*r.shirt_recall:.1f} & {r.best_epoch}/{r.stop_epoch} & {r.parameters:,} \\\\" for r in evidence["cnn_candidates"].itertuples()]
(GENERATED / "ablations.tex").write_text(finish(rows))
rows = []
per_class = evidence["per_class"]
for label in range(10):
    c = per_class[(per_class.run_id == CNN_ID) & (per_class.stage == "test") & (per_class.label == label)].iloc[0]
    s = per_class[(per_class.run_id == SVM_ID) & (per_class.stage == "test") & (per_class.label == label)].iloc[0]
    rows.append(f"{c.category} & {100*s.recall:.1f} & {100*c.recall:.1f} & {100*(c.recall-s.recall):+.1f} \\\\")
(GENERATED / "recalls.tex").write_text(finish(rows))
rows = []
for row in comparison.itertuples():
    timed = evidence["timing"]["measurements"][row.row]["batch128_amortized"]
    rows.append(f"{row.model} & {row.ms_per_image:.3f} & {timed['min_ms_per_image']:.3f}--{timed['max_ms_per_image']:.3f} \\\\")
(GENERATED / "timing.tex").write_text(finish(rows))
macros = {
    "CnnAccuracy": f"{100*cnn.test_accuracy:.2f}\\%", "CnnFone": f"{cnn.test_macro_f1:.4f}",
    "SvmAccuracy": f"{100*svm.test_accuracy:.2f}\\%", "SvmFone": f"{svm.test_macro_f1:.4f}",
    "AccuracyGain": f"{100*(cnn.test_accuracy-svm.test_accuracy):.2f}",
    "FoneGain": f"{cnn.test_macro_f1-svm.test_macro_f1:.4f}",
    "CnnShirt": f"{100*cnn.shirt_recall:.1f}\\%", "SvmShirt": f"{100*svm.shirt_recall:.1f}\\%",
    "CnnTime": f"{cnn.ms_per_image:.3f}", "SvmTime": f"{svm.ms_per_image:.3f}",
    "SpeedRatio": f"{svm.ms_per_image/cnn.ms_per_image:.1f}",
    "AccCiLower": f"{100*uncertainty['accuracy_delta_ci95'][0]:.2f}",
    "AccCiUpper": f"{100*uncertainty['accuracy_delta_ci95'][1]:.2f}",
}
(GENERATED / "values.tex").write_text(finish(["\\newcommand{\\" + name + "}{" + value + "}" for name, value in macros.items()]))
for stem in ["comparison", "learning", "ablations", "recall", "hybrid", "errors"]:
    shutil.copyfile(ROOT / "figures/phase2/integration" / f"{stem}.pdf", FIGURES / f"{stem}.pdf")
shutil.copyfile(ROOT / "report/references.bib", REPORT / "references.bib")
with (REPORT / "references.bib").open("a") as stream:
    stream.write(r'''
@article{srivastava2014dropout,
 author={Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan Salakhutdinov},
 title={Dropout: A Simple Way to Prevent Neural Networks from Overfitting},
 journal={Journal of Machine Learning Research}, year={2014}, volume={15}, pages={1929--1958},
 url={https://www.jmlr.org/papers/v15/srivastava14a.html}}
@inproceedings{loshchilov2019adamw,
 author={Ilya Loshchilov and Frank Hutter}, title={Decoupled Weight Decay Regularization},
 booktitle={International Conference on Learning Representations}, year={2019},
 url={https://arxiv.org/abs/1711.05101}}
''')
# Private author details follow the same user-supplied local record as Phase 1.
private_author = ROOT / "report/author_details.local.tex"
if private_author.exists():
    shutil.copyfile(private_author, REPORT / "author_details.local.tex")
    (REPORT / "submission.tex").write_text("\\def\\IncludePrivateAuthors{1}\n\\input{main.tex}\n")
print("Prepared measured Phase 2 report tables, macros, figures and references.")

# Inline generated alignment rows for the IEEE table environment. Markers let
# later runs refresh measured rows without replacing the report's narrative.
import re
main_path = REPORT / "main.tex"
source = main_path.read_text()
for stem in ["comparison", "ablations", "recalls", "timing"]:
    begin, end = f"% BEGIN GENERATED {stem}", f"% END GENERATED {stem}"
    block = begin + "\n" + (GENERATED / f"{stem}.tex").read_text().strip() + "\n" + end
    include = "\\input{generated/" + stem + ".tex}"
    if include in source:
        source = source.replace(include, block)
    elif begin in source:
        source = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda match: block, source, flags=re.S)
main_path.write_text(source)

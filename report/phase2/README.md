# Phase 2 IEEE/LaTeX report

`main.tex` is the public main source and `main.pdf` is its compiled output.
`generated/` contains measured tables and macros. `figures/` contains portable
publication figures. `references.bib` contains the cited research.

From the repository root, regenerate measured evidence and compile with Tectonic.

```bash
python -m clothing_ml.phase2.reporting
python scripts/build_phase2_report.py
work/tectonic --keep-logs --outdir report/phase2 report/phase2/main.tex
```

Use an installed Tectonic or another IEEEtran-compatible LaTeX setup if the
ignored local compiler is absent. The report folder can be uploaded to Overleaf
with `main.tex` as the main document.

The git-ignored `author_details.local.tex`, `submission.tex` and `submission.pdf`
provide the user-supplied student IDs for private submission. The final private
source ZIP includes those files. Keep the public source on `main.tex`.

# Report build and submission copies

- `main.tex`: IEEE-format source with measured tables embedded by the generator.
- `references.bib`: checked research and implementation references.
- `figures/`: publication-sized, source-backed figures.
- `generated/values.tex`: measured values/macros generated from frozen evidence.
- `main.pdf`: public copy, first names and Section A, without student IDs.
- `submission.pdf`: local private copy with all supplied roll numbers; git-ignored.

The public and private copies contain the same experiments and results. Student
IDs are supplied through git-ignored `private/student_details.json` and
`author_details.local.tex`. `submission.tex` activates private author details.
Do not upload those private files to the public repository inadvertently.

## Regenerate text and report figures

From the repository root, after the saved final results exist:

```bash
python scripts/build_report.py
```

The script reads the checked-in full-profile evidence. It does not fit models,
retune settings or make new test-selection decisions. It also updates the
presentation content outline.

## Compile

Use an installed Tectonic compiler from its official project, or upload the
report folder to Overleaf and use its LaTeX compiler with `main.tex` as the main
document. The Tectonic binary used locally is a build dependency under `work/`
and is not committed. The local commands from the repository root are:

```bash
work/tectonic --keep-logs --outdir report report/main.tex
work/tectonic --keep-logs --outdir report report/submission.tex
```

For the private Overleaf submission copy, include `author_details.local.tex`
and `submission.tex` from your local checkout or private source ZIP, and select
`submission.tex` as the main file. For the public copy select `main.tex`.
Keep `generated/values.tex`, `figures/`, and `references.bib` with either copy.

The desktop editor also compiles the project with remembered main file
`main.tex`. Open the private PDF directly when checking student IDs.

The faculty's exact template, page limit, affiliation wording and evaluation
date have not been supplied. The provided draft uses the standard IEEEtran
conference class and real measured results. All members must review its
technical content and the assistance/contribution statement before submission.

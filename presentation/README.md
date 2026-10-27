# Phase 1 presentation

Open `Phase_1_Clothing_Category_Recognition.pptx` to edit or present the deck.
The PDF is a portable copy. Both have 22 slides: 18 main slides and four backup
slides. Speaker notes are embedded in the PowerPoint and also provided in
`speaker_notes.md`. Adjust pacing to the faculty time limit.

## Speaking assignments

| Slides | Presenter | Subject |
|---|---|---|
| 1–3 | Nakul | Context, research question and team responsibilities |
| 4–5 | Sudeshhna | Dataset, checks and partitions |
| 6–7 | Puneet | Features, Logistic Regression and kNN |
| 8–9 | Samarth | Random Forest, SVM and validation selection |
| 10–15 | Neelanshu | Metrics, results, errors and inference time |
| 16–18 | Nakul | Reproducibility, conclusions, Phase 2 and references |
| 19–22 | Relevant member / all | Backup results, settings, complexity and viva/demo |

## Evidence and editing

`evidence.json` contains values derived from the frozen full-profile result
files. `assets/` contains original dataset examples, not synthetic illustrations.
The chart series, tables and text remain editable. Chart F1 values round to six
decimal places for workbook compatibility. The source CSVs retain full precision.
The PDF preserves the final slide layout, while the PowerPoint retains notes.
Public copies use first names and omit roll numbers.

`python scripts/prepare_presentation.py` refreshes the deck evidence and extracts
the original sample images from existing checksum-verified data. It does not
train models or change selection. `scripts/build_slides.mjs` is the authoring
recipe for the Artifact Session JavaScript environment, where `artifactSession`
and `nodeRepl` are provided. It is not a standalone Node command. The live
presentation is the editable authoring source. Native chart workbooks in the
PowerPoint contain the displayed chart data as literal values.

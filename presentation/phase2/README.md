# Phase 2 presentation

Open `Phase_2_Clothing_Category_Recognition.pptx` in PowerPoint or compatible slide
software. It contains 20 slides, 10 native tables and five native charts on four
slides. All 20 slides have short Hinglish speaker notes with presenter assignments
and source references. `speaker_notes.md` provides the same notes for rehearsal.
The table and chart owners were validated, and the exported file passed a fresh
Artifact Tool import. Visual review used the rendered slide PNGs.

`evidence.json` is derived from the reviewed experiment records. The three example
images retain their original dataset IDs. Native charts use six-decimal numeric
snapshots, with editable embedded workbooks generated from those exact values.
Those workbooks are new data snapshots; model and experiment source files remain
in the repository.

## Rebuild in a Codex workspace with Artifact Tool

From the repository root, set `RUNTIME_NODE_MODULES` to the bundled runtime's
`node_modules` directory, `PRESENTATIONS_SKILL` to the installed presentations
skill folder, and `PYTHON_EXECUTABLE` to a Python executable with openpyxl/lxml.
Use the bundled Node executable if it is not on PATH.

```bash
python scripts/prepare_phase2_slides.py
node scripts/build_phase2_slides.mjs
node scripts/finalize_phase2_slides.mjs
```

The authoring runtime is specific to Codex. Editing the final PowerPoint does not
require that runtime. Rendering candidates and validation receipts stay under
ignored `work/phase2_integration/`. To preserve a reviewed deck during a rebuild,
set `PHASE2_PPTX_OUTPUT` to a new absolute destination before finalizing.

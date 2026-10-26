# Nakul: start here

## What you can use immediately

Your Desktop folder is the project checkout. The repository contains the data
loader, saved splits, both feature representations, all four classical model
families, validation selection, guarded final evaluation, notebook, figures and
report source. See README for the current executed result locations.

Read the executed master notebook first. It explains the experiment using real
outputs, and can be rerun without retraining all models. Training is explicit
through the CLI because it is slower and final testing must remain controlled.

## Your review path (about 2-3 focused hours initially)

1. **20 minutes: problem and data.** Read README and `docs/protocol.md`. Open the
   training gallery. Explain an image, feature vector, label and original ID.
2. **25 minutes: follow one prediction.** Open `src/clothing_ml/features.py`.
   Trace a raw image through normalization and either feature representation.
   Open `models.py` and explain what its Pipeline learns from training data.
3. **25 minutes: understand the experiment.** Read `experiments.py` from train
   to freeze to final_test. Explain why test data is loaded only at the end.
   Identify the primary metric and what makes a finalist eligible.
4. **25 minutes: verify results.** Pick one prediction CSV and recompute accuracy.
   Check the per-class recall table and read the confusion matrix in the right
   direction. Explain one example where HOG helps and one where it fails.
5. **25 minutes: review writing.** Read the literature draft and report. Verify
   that its numbers match the executed evidence. Fill only real course details.
6. **20 minutes: rehearse.** Give a five-minute explanation without reading the
   notebook aloud. Have a teammate ask a question about a different component.

## The entire project in plain language

We have labeled clothing pictures. We convert each picture to numbers, train
several decision rules, and compare them on the same held-out examples. We use
validation examples to make choices and save the test examples for the final
measurement. Pixels preserve raw brightness; HOG summarizes local edge layout.
Neither path learns a CNN feature extractor in this phase.

LR learns linear class scores, kNN compares to stored examples, RF combines
randomized trees, and SVM uses margin-based nonlinear decisions. A higher test
score does not automatically imply a smaller, faster or more deployable model.

## Your viva questions

- Why is the split 48,000/12,000/10,000? Where did each partition come from?
- What is the budget development stage, and how is it separated from the full run?
- Why can HOG have 1,296 values when pixels have only 784?
- What does standardization learn? Where is it fitted in the code?
- Why is kNN's low fit time not a complete efficiency argument?
- How is macro F1 calculated? Which class has the weakest measured recall?
- Which model was selected on validation, and was that decision changed after test?
- What does the inference timer include? Is it single-request latency?
- What part of the literature review is directly relevant to Phase 1?
- What did AI prepare, and what have you personally reviewed or changed?

## What still needs humans

All five members must review their assigned part and record actual contributions.
No generated document can demonstrate individual understanding for the viva.
Confirm the faculty's evaluation date, report template/page limit, AI-use policy,
section, roll numbers and affiliation details. These have not been supplied.
Do not invent them or claim completed teammate contributions.


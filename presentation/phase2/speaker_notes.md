# Phase 2 presenter notes

## 1. Automatic clothing category recognition

Presenter: Nakul

Phase 1 mein fixed pixels aur HOG compare hue the. Phase 2 mein learned CNN features ka result dekhte hain. Same dataset splits use hue hain. CNN aggregate score mein better raha, lekin Shirt recall aur hybrid ka result mixed hai.

Sources
- report/phase2/main.tex

## 2. Learned features and similar garments

Presenter: Nakul

Main question tha ki learned filters similar garments ko better separate kar paate hain ya nahi. LeCun ka paper CNN design ko motivate karta hai. Dropout aur AdamW ke papers regularization choices explain karte hain. Numbers apne saved experiments se aaye hain.

Sources
- docs/phase2_literature_review.md
- https://bottou.org/papers/lecun-98h
- https://www.jmlr.org/papers/v15/srivastava14a.html
- https://arxiv.org/abs/1711.05101

## 3. Shared dataset and split membership

Presenter: Sudeshhna

48,000 fit, 12,000 validation aur 10,000 official test images same rakhe hain. CNN ko 28 by 28 grayscale input milta hai, 255 se ek baar divide karke. Original IDs preserve hain, isliye models ki predictions image-wise compare ho sakti hain.

Sources
- data/full_indices.npz
- data/provenance.json
- results/phase2/data_audit/data_audit.json

## 4. Selected CNN architecture

Presenter: Puneet

Selected CNN mein three convolution blocks hain, channels 32, 64 aur 128. Har block ke baad ReLU aur pooling hai. Dense layer 64-value embedding banati hai. Total 167,114 parameters hain. Third pooling dense input ko chhota kar deti hai.

Sources
- src/clothing_ml/phase2/models.py
- results/phase2/full/runs/full_cnn3_s42_80a2edee/record.json

## 5. Training and best-checkpoint selection

Presenter: Samarth

Training mein AdamW, cross-entropy aur batch 128 use hua. Validation macro F1 se best checkpoint select hota hai. CNN-3 ka best epoch 22 tha, aur training epoch 25 tak gayi. Final predictions ke liye best checkpoint reload hua.

Sources
- results/phase2/full/runs/full_cnn3_s42_80a2edee/history.csv
- docs/phase2_training_protocol.md

## 6. Validation ablations

Presenter: Samarth

Dropout hataane aur har image shift karne se validation F1 gira. CNN-3 reference se improve hua. Later 50 percent shift run ka F1 thoda badha, par Shirt recall gira. Woh later validation-only ablation tha, frozen model selection ka part nahi tha.

Sources
- results/phase2/integration/cnn_candidates.csv
- results/phase2/data_audit/augmentation_ablation.csv

## 7. Validation choice and frozen test rows

Presenter: Samarth

CNN validation F1 se select hua, phir exact artifacts aur LR controls freeze hue. Uske baad official test score hua. Phase 1 mein public test already dekha gaya tha. Phase 2 choices validation par hui, aur same benchmark par comparison report karte hain.

Sources
- results/phase2/full/cnn_selection.json
- results/phase2/full/freeze.json
- results/phase2/full/test_access_log.md

## 8. Frozen official-test macro F1

Presenter: Neelanshu

Selected CNN ki test accuracy 92.14 percent aur macro F1 0.9207 hai. HOG plus SVM se accuracy 1.23 percentage points badhi. Embedding-only LR control ka aggregate score thoda aur high hai. Yeh saare rows pehle se frozen comparison mein included the.

Sources
- results/phase2/integration/comparison.csv

## 9. Paired CNN gain over HOG + SVM

Presenter: Neelanshu

Paired bootstrap mein same image IDs ko saath resample kiya. Accuracy gain ka 95 percent interval 0.73 se 1.75 points hai. Yeh benchmark-image sampling ka interval hai. Training seed variation measure nahi hui, kyunki primary runs ek seed par hain.

Sources
- results/phase2/full/evaluation/cnn_vs_svm_deltas.json
- results/phase2/full/evaluation/paired_summary.json

## 10. Category recall still matters

Presenter: Neelanshu

Most categories mein CNN recall improve karta hai. Shirt exception hai: 73.3 se 71.7 percent. Bag aur footwear ka recall relatively strong hai. Isliye overall accuracy ke saath har class ka result dekhna zaroori hai.

Sources
- results/phase2/integration/per_class.csv

## 11. Shirt precision rises while recall falls

Presenter: Neelanshu

CNN Shirt ko predict karne mein zyada selective hua. Shirt precision aur F1 badhe, lekin recall gira. CNN 134 true Shirts ko T-shirt bola, SVM ne 103 ko. Reverse direction mein T-shirt ko Shirt bolne ki errors kam hui.

Sources
- results/phase2/full/evaluation/per_class_metrics.csv
- results/phase2/full/evaluation/label_pair_counts.csv

## 12. Paired upper-body prediction examples

Presenter: Neelanshu

Same test images par paired mistakes compare ki hain. Pehla example sirf CNN sahi karta hai, doosra sirf SVM, aur teesra dono se wrong hota hai. Examples group ke first sorted IDs se hain. Visible shape se possible reasons suggest kar sakte hain.

Sources
- results/phase2/full/evaluation/figure_example_ids.csv

## 13. Hybrid feature fusion and controls

Presenter: Samarth

Hybrid mein 64 learned embedding values aur 1,296 HOG values join hote hain. Phir LR train hota hai. Embedding-only aur HOG-only LR controls bhi hain. Same scaling aur C grid se check hota hai ki features combine karna useful tha ya nahi.

Sources
- src/clothing_ml/phase2/fusion.py
- results/phase2/full/fusion/fusion_candidates.csv

## 14. Fusion adds no aggregate improvement here

Presenter: Samarth

Fusion validation F1 0.9261 hai, embedding-only LR ka 0.9319. Test par bhi fusion thoda below hai. Is experiment mein aggregate hybrid improvement support nahi hota. Higher Shirt recall test ke baad dikha, toh usko exploratory trade-off kehte hain.

Sources
- results/phase2/full/fusion/fusion_candidates.csv
- results/phase2/integration/comparison.csv

## 15. Recorded CPU inference cost

Presenter: Neelanshu

Same CPU setup mein CNN ka cost 0.312 milliseconds per image tha, SVM ka 19.748. Yeh batch 128, five repeats ka median hai, input conversion aur features included hain. SVM refit ne saari saved predictions match ki thi, phir timing hui.

Sources
- results/phase2/full/evaluation/timing.json

## 16. Model size uses model-specific units

Presenter: Nakul

CNN ke 167,114 trainable values hain. SVM ka count stored decision arrays ka hai, isliye units table mein clear hain. Hybrid ko frozen encoder ke saath LR head bhi ship karna padta hai. File sizes compression aur format par bhi depend karte hain.

Sources
- results/phase2/full/evaluation/complexity.json

## 17. Integrated and reproducible evidence

Presenter: Nakul

Integration mein 31 prediction files official labels ke against check hui. IDs, metrics aur hashes match hain. Selected checkpoint ne 12,000 validation predictions reproduce ki. Notebook, report aur slides same reviewed evidence use karte hain.

Sources
- results/phase2/integration/audit.json
- docs/phase2_integration_nakul.md

## 18. Conclusions and measured limits

Presenter: Nakul

Overall learned features se accuracy aur macro F1 improve hue, aur recorded CPU cost kam hui. Shirt confusion abhi bacha hai. Fusion ka gain support nahi hua. One seed, small grayscale images aur one-machine timing ki wajah se conclusions isi experiment tak bounded hain.

Sources
- report/phase2/main.tex
- results/phase2/integration/comparison.csv

## 19. Selected references

Presenter: Nakul

Yeh main research sources hain. Full citations report aur literature notes mein hain. Dataset, CNN design, dropout aur AdamW ke ideas yahan se aaye. Thank you, ab questions le sakte hain.

Sources
- docs/phase2_literature_review.md
- report/phase2/references.bib

## 20. Full results and offline demo

Presenter: All five members

Offline demo mein notebook kholkar original test ID 40 dekh sakte hain. True class Shirt hai, CNN correct hai aur SVM T-shirt bolta hai. Phir prediction CSV se metric trace karenge. Viva mein split, checkpoint rule aur apne result ka limitation explain karna hai.

Sources
- results/phase2/integration/comparison.csv
- notebooks/01_phase2_master.ipynb

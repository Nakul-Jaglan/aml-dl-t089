# Phase 1 presenter notes

Slides 1–18 form the main presentation. Slides 19–22 are backup slides.
Short spoken notes use Roman-script Hinglish. Sources are reference material, not spoken lines.
Speaking assignments follow the team responsibilities. Adjust pacing to the faculty time limit.

## 1. Automatic clothing category recognition

Presenter: **Nakul**

Is project mein clothing ki image se uski category predict karni hai, jaise shirt, trouser ya sneaker. Input 28 by 28 grayscale image hai aur total 10 categories hain. Abhi Phase 1 mein classical ML models ka comparison hai. CNN Phase 2 mein aayega.

Sources:
- docs/course_instructions.md
- data/provenance.json

## 2. Research question and prior work

Presenter: **Nakul**

Main question yeh hai ki raw pixels ke comparison mein HOG ke edge features kitne useful hain. Aur kya har classifier ko same benefit milta hai? Fashion-MNIST dataset ka base hai, aur HOG ka idea Dalal aur Triggs ke paper se aata hai.

Sources:
- https://arxiv.org/abs/1708.07747
- https://doi.org/10.1109/CVPR.2005.177
- https://doi.org/10.1023/A:1010933404324
- https://doi.org/10.1007/BF00994018
- docs/literature_review.md

## 3. Team responsibilities

Presenter: **Nakul**

Team mein responsibilities aise divided hain: main integration aur reporting explain karunga, Sudeshhna data aur splits, Puneet features aur LR-kNN, Samarth RF-SVM aur selection, aur Neelanshu results aur errors. Ab dataset ke liye Sudeshhna continue karegi.

Sources:
- docs/contributions.md

## 4. Fashion-MNIST dataset

Presenter: **Sudeshhna**

Fashion-MNIST mein 60,000 training aur 10,000 test images hain. Har image 28 by 28 grayscale hai. Yahan 10 categories ke training examples dikh rahe hain. Classes balanced hain, lekin real-world photos mein background aur lighting alag ho sakti hai.

Sources:
- https://github.com/zalandoresearch/fashion-mnist
- data/training_audit.json
- figures/eda/gallery_ids.json

## 5. Data checks and fair partitions

Presenter: **Sudeshhna**

60,000 training images ko 48,000 fit aur 12,000 validation mein divide kiya hai. Official test ke 10,000 images alag rakhe hain. Shape, pixel range aur exact duplicates ke checks pass hue. Scaler sirf fit data se seekhta hai, taaki validation ki information leak na ho.

Sources:
- data/training_audit.json
- data/provenance.json
- data/split_indices.npz
- src/clothing_ml/data.py

## 6. Pixels and HOG features

Presenter: **Puneet**

Pehle pixel values ko 255 se divide karke zero se one range mein laate hain. Flatten karne par 784 features milte hain. HOG local edge directions capture karta hai. In settings mein 1,296 features bante hain, matlab HOG vector pixels se bada hai.

Sources:
- src/clothing_ml/features.py
- figures/eda/hog_example.png
- https://doi.org/10.1109/CVPR.2005.177

## 7. Logistic Regression and kNN

Presenter: **Puneet**

Logistic Regression har category ke liye linear score seekhta hai. kNN nearby training examples ke votes se prediction karta hai. LR mein dono features scale hote hain. kNN mein HOG scale hota hai, pixels sirf normalize hote hain. kNN ka prediction stored examples par depend karta hai.

Sources:
- src/clothing_ml/models.py
- docs/protocol.md
- results/full/freeze.json

## 8. Random Forest and RBF SVM

Presenter: **Samarth**

Random Forest mein 200 decision trees ke results combine hote hain. RBF SVM nonlinear patterns handle karta hai. SVM mein C error penalty ko control karta hai aur gamma similarity ka spread. SVM ke inputs scale hote hain, Random Forest ke nahi.

Sources:
- src/clothing_ml/models.py
- results/full/freeze.json
- https://doi.org/10.1023/A:1010933404324
- https://doi.org/10.1007/BF00994018

## 9. Model selection before test access

Presenter: **Samarth**

Pehle small smoke run se pipeline check hoti hai. Phir 12,000 fit images par har combination ke do settings compare hote hain. Selected settings se saare eight models 48,000 images par fit hote hain. Validation macro F1 se winner freeze hota hai, uske baad official test evaluate hota hai.

Sources:
- docs/protocol.md
- results/budget/validation_candidates.csv
- results/full/freeze.json
- results/full/test_receipt.json

## 10. Evaluation measures

Presenter: **Neelanshu**

Accuracy batati hai total predictions mein se kitni correct hain. Recall batata hai kisi class ke actual examples mein se kitne identify hue. Macro F1 har class ke F1 ko equal weight deta hai. Saath mein prediction time aur model size bhi compare karte hain.

Sources:
- src/clothing_ml/evaluate.py
- tests/test_integrity.py
- results/full/per_class_recall.csv

## 11. HOG helps three classifier families

Presenter: **Neelanshu**

Chart mein HOG se LR, kNN aur SVM ka macro F1 improve hua, lekin Random Forest ka thoda kam hua. Validation-selected HOG plus RBF SVM ki test accuracy 90.91 percent aur macro F1 0.9090 hai. Yeh result isi split aur search setup ke liye hai.

Sources:
- results/full/test_summary.csv
- results/full/freeze.json
- presentation/evidence.json

## 12. Shirt recall remains the weakest

Presenter: **Neelanshu**

Overall accuracy achhi hai, lekin Shirt sabse weak category hai. Uska recall 73.3 percent hai, matlab 1,000 actual Shirts mein se 733 correct identify hue. Bag ka recall 98.3 percent hai. Isliye category-wise result dekhna bhi zaroori hai.

Sources:
- results/full/full_hog_svm_3fd9d87a_test_metrics.json
- results/full/per_class_recall.csv

## 13. Confusion among upper-body categories

Presenter: **Neelanshu**

Table mein rows actual labels hain aur columns predictions. 104 T-shirt images Shirt predict hui, aur 103 Shirts T-shirt predict hui. Shirt ka Pullover aur Coat se bhi confusion hai. Other column remaining six categories ko combine karta hai, isliye har row ka total 1,000 hai.

Sources:
- results/full/full_hog_svm_3fd9d87a_test_metrics.json
- presentation/evidence.json

## 14. Paired SVM prediction examples

Presenter: **Neelanshu**

Same test images par dono SVM predictions compare hain. Pehle do examples mein sirf HOG sahi hai, last do mein dono wrong hain. Examples group ke first IDs se choose hue hain. Yeh errors samjhate hain, lekin inse saare errors ka proportion ya exact cause prove nahi hota.

Sources:
- results/full/error_example_ids.csv
- report/figures/error_examples.png
- src/clothing_ml/plots.py

## 15. Inference cost changes the trade-off

Presenter: **Neelanshu**

HOG plus SVM ki accuracy better hai, lekin is machine par end-to-end cost lagbhag 21.7 milliseconds per image hai. Timing mein feature extraction included hai, loading nahi. Yeh batch 128 aur five repeats ka median hai, single-image response time nahi.

Sources:
- results/full/timing.json
- results/full/test_summary.csv

## 16. Reproducible Phase 1 evidence

Presenter: **Nakul**

Repo mein saved splits, prediction files, model settings aur executed notebook available hain. Inse reported numbers trace aur recheck ho sakte hain. Eight integrity tests bhi pass hue hain. Fresh run ke results alag folder mein save hote hain, taaki original experiment evidence preserve rahe.

Sources:
- results/full/evidence_audit.json
- tests/test_integrity.py
- .github/workflows/checks.yml
- notebooks/00_phase1_master.ipynb
- docs/protocol.md

## 17. Conclusions and Phase 2 proposal

Presenter: **Nakul**

Phase 1 mein HOG plus RBF SVM strongest selected model raha. Lekin Shirt confusion aur inference cost abhi limitations hain. Phase 2 mein same splits par small CNN compare karna hai. CNN better hoga ya nahi, woh experiment ke baad hi pata chalega.

Sources:
- results/full/test_summary.csv
- results/full/full_hog_svm_3fd9d87a_test_metrics.json
- docs/course_instructions.md
- report/main.tex

## 18. Selected references

Presenter: **Nakul**

Yeh main references hain. Fashion-MNIST dataset ke liye Xiao aur colleagues, HOG ke liye Dalal aur Triggs, aur model background ke liye Random Forest aur SVM papers use hue hain. Full references report mein hain. Thank you, ab questions le sakte hain.

Sources:
- docs/literature_review.md
- report/references.bib

## 19. All eight finalists

Presenter: **Samarth and Neelanshu** (backup)

Yeh backup table saare eight finalists ke validation F1, test accuracy aur test F1 dikhata hai. Har model ka final fit size same hai. HOG plus SVM validation par select hua tha, test result dekhkar nahi. Exact values source CSVs mein available hain.

Sources:
- results/full/validation_summary.csv
- results/full/test_summary.csv

## 20. Selected settings and search scope

Presenter: **Samarth** (backup)

Yahan selected settings aur scaling policy hain. Search chhota tha: LR aur SVM ke liye C, kNN ke liye k, aur Forest ke liye depth compare hui. Isliye in results ko har model ki best possible performance nahi keh sakte.

Sources:
- results/full/freeze.json
- results/budget/validation_candidates.csv
- src/clothing_ml/models.py

## 21. Model size and complexity

Presenter: **Nakul** (backup)

Model complexity ka meaning har family mein alag hai. LR mein weights, kNN mein stored features, Forest mein nodes, aur SVM mein decision arrays count hue hain. MiB column saved file size hai, RAM usage nahi. In counts ko CNN parameter count se directly compare nahi karna chahiye.

Sources:
- results/full/complexity.json
- results/full/freeze.json
- results/full/test_summary.csv

## 22. Viva and saved-prediction demo

Presenter: **All five members** (backup)

Demo mein saved notebook kholkar test ID zero dikha sakte hain. Actual label aur selected model ki prediction dono Ankle boot hain. Phir saved predictions se accuracy recalculate kar sakte hain. CNN ka result abhi available nahi hai, toh uske performance ka claim nahi karenge.

Sources:
- notebooks/00_phase1_master.ipynb
- notebooks/00_phase1_master.html
- results/full/predictions/full_hog_svm_3fd9d87a_test.csv.gz
- docs/START_HERE_NAKUL.md

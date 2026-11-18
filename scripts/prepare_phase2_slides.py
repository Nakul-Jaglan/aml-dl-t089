"""Assemble the final slide evidence and concise Hinglish notes from reviewed runs."""
from pathlib import Path
import json

import pandas as pd
from PIL import Image

from clothing_ml.data import CLASS_NAMES, load_official
from clothing_ml.phase2.reporting import FULL, CNN_ID, SVM_ID, reviewed_evidence
from clothing_ml.utils import ROOT, read_json

folder = ROOT / "presentation/phase2"
(folder / "assets").mkdir(parents=True, exist_ok=True)
evidence = reviewed_evidence()
comparison = evidence["comparison"]
per_class = evidence["per_class"]
records = evidence["cnn_records"]
history = pd.DataFrame(records[CNN_ID]["history"])
candidates = evidence["cnn_candidates"]
examples = pd.read_csv(FULL / "evaluation/figure_example_ids.csv")
images, _ = load_official("test")
example_rows = []
for group in ["Only CNN correct", "Only HOG + SVM correct", "Both wrong"]:
    row = examples[(examples.panel == "upper_body") & (examples.group == group)].iloc[0].to_dict()
    Image.fromarray(images[int(row["sample_id"]) ]).save(folder / "assets" / f"test_{row['sample_id']}.png")
    example_rows.append(row)

architecture = [
    ["Stage", "Output shape", "Trainable values"],
    ["Conv 1 + ReLU + pool", "32 x 14 x 14", "320"],
    ["Conv 2 + ReLU + pool", "64 x 7 x 7", "18,496"],
    ["Conv 3 + ReLU + pool", "128 x 3 x 3", "73,856"],
    ["Dense + ReLU", "64", "73,792"],
    ["Dropout + output", "10 logits", "650"],
    ["Total", "CNN-3", "167,114"],
]
result_rows = [["Pipeline", "Val F1", "Test accuracy", "Test F1"]]
for row in comparison.itertuples():
    result_rows.append([row.model, f"{row.val_macro_f1:.4f}", f"{row.test_accuracy:.2%}", f"{row.test_macro_f1:.4f}"])
ablation_rows = [["Configuration", "Val F1", "Shirt recall", "Best / final"]]
for row in candidates.itertuples():
    name = {"cnn2":"CNN-2", "cnn2_p0":"No dropout", "cnn2_aug":"Always shift", "cnn3":"CNN-3", "cnn2_aug_p50":"50% shift (later)"}[row.name]
    ablation_rows.append([name, f"{row.val_macro_f1:.4f}", f"{row.shirt_recall:.1%}", f"{row.best_epoch} / {row.stop_epoch}"])
recall = {}
shirt_rows = [["Pipeline", "Precision", "Recall", "F1"]]
for rid, name in [(SVM_ID,"HOG + SVM"),(CNN_ID,"CNN-3")]:
    sub = per_class[(per_class.stage=="test") & (per_class.run_id==rid)].sort_values("label")
    recall[name] = sub.recall.tolist()
    row = sub[sub.label==6].iloc[0]
    shirt_rows.append([name,f"{row.precision:.1%}", f"{row.recall:.1%}", f"{row.f1:.3f}"])
raw_complexity = read_json(FULL / "evaluation/complexity.json")
complexity = [["Pipeline", "Quantity", "Meaning"]]
for row in raw_complexity:
    if row["row"]=="selected_cnn": label, meaning = "CNN-3", "Trainable parameters"
    elif row["row"]=="hog_svm": label, meaning = "HOG + SVM", "Stored decision scalars"
    else:
        label = {"emb_lr":"Embedding + LR","hog_lr":"HOG + LR","fusion_lr":"Fusion + LR"}[row["row"]]
        meaning = "LR weights and biases"
        if row.get("frozen_encoder_scalars_shipped",0): meaning += "; encoder 166,464"
    complexity.append([label,f"{row['value']:,}",meaning])
data = {"class_names":CLASS_NAMES,"architecture":architecture,"results_table":result_rows,
        "ablation_table":ablation_rows,"recall":recall,"shirt_table":shirt_rows,"complexity_table":complexity,
        "comparison":comparison.to_dict("records"),"history":history.to_dict("records"),"examples":example_rows,
        "paired_groups":read_json(FULL / "evaluation/paired_summary.json")["groups"],
        "uncertainty":read_json(FULL / "evaluation/cnn_vs_svm_deltas.json"),
        "freeze":evidence["freeze"]["manifest_sha256"],"audit_count":evidence["audit_count"]}
(folder / "evidence.json").write_text(json.dumps(data,indent=2)+"\n")
notes = [
("Nakul", "Phase 1 mein fixed pixels aur HOG compare hue the. Phase 2 mein learned CNN features ka result dekhte hain. Same dataset splits use hue hain. CNN aggregate score mein better raha, lekin Shirt recall aur hybrid ka result mixed hai."),
("Nakul", "Main question tha ki learned filters similar garments ko better separate kar paate hain ya nahi. LeCun ka paper CNN design ko motivate karta hai. Dropout aur AdamW ke papers regularization choices explain karte hain. Numbers apne saved experiments se aaye hain."),
("Sudeshhna", "48,000 fit, 12,000 validation aur 10,000 official test images same rakhe hain. CNN ko 28 by 28 grayscale input milta hai, 255 se ek baar divide karke. Original IDs preserve hain, isliye models ki predictions image-wise compare ho sakti hain."),
("Puneet", "Selected CNN mein three convolution blocks hain, channels 32, 64 aur 128. Har block ke baad ReLU aur pooling hai. Dense layer 64-value embedding banati hai. Total 167,114 parameters hain. Third pooling dense input ko chhota kar deti hai."),
("Samarth", "Training mein AdamW, cross-entropy aur batch 128 use hua. Validation macro F1 se best checkpoint select hota hai. CNN-3 ka best epoch 22 tha, aur training epoch 25 tak gayi. Final predictions ke liye best checkpoint reload hua."),
("Samarth", "Dropout hataane aur har image shift karne se validation F1 gira. CNN-3 reference se improve hua. Later 50 percent shift run ka F1 thoda badha, par Shirt recall gira. Woh later validation-only ablation tha, frozen model selection ka part nahi tha."),
("Samarth", "CNN validation F1 se select hua, phir exact artifacts aur LR controls freeze hue. Uske baad official test score hua. Phase 1 mein public test already dekha gaya tha. Phase 2 choices validation par hui, aur same benchmark par comparison report karte hain."),
("Neelanshu", "Selected CNN ki test accuracy 92.14 percent aur macro F1 0.9207 hai. HOG plus SVM se accuracy 1.23 percentage points badhi. Embedding-only LR control ka aggregate score thoda aur high hai. Yeh saare rows pehle se frozen comparison mein included the."),
("Neelanshu", "Paired bootstrap mein same image IDs ko saath resample kiya. Accuracy gain ka 95 percent interval 0.73 se 1.75 points hai. Yeh benchmark-image sampling ka interval hai. Training seed variation measure nahi hui, kyunki primary runs ek seed par hain."),
("Neelanshu", "Most categories mein CNN recall improve karta hai. Shirt exception hai: 73.3 se 71.7 percent. Bag aur footwear ka recall relatively strong hai. Isliye overall accuracy ke saath har class ka result dekhna zaroori hai."),
("Neelanshu", "CNN Shirt ko predict karne mein zyada selective hua. Shirt precision aur F1 badhe, lekin recall gira. CNN 134 true Shirts ko T-shirt bola, SVM ne 103 ko. Reverse direction mein T-shirt ko Shirt bolne ki errors kam hui."),
("Neelanshu", "Same test images par paired mistakes compare ki hain. Pehla example sirf CNN sahi karta hai, doosra sirf SVM, aur teesra dono se wrong hota hai. Examples group ke first sorted IDs se hain. Visible shape se possible reasons suggest kar sakte hain."),
("Samarth", "Hybrid mein 64 learned embedding values aur 1,296 HOG values join hote hain. Phir LR train hota hai. Embedding-only aur HOG-only LR controls bhi hain. Same scaling aur C grid se check hota hai ki features combine karna useful tha ya nahi."),
("Samarth", "Fusion validation F1 0.9261 hai, embedding-only LR ka 0.9319. Test par bhi fusion thoda below hai. Is experiment mein aggregate hybrid improvement support nahi hota. Higher Shirt recall test ke baad dikha, toh usko exploratory trade-off kehte hain."),
("Neelanshu", "Same CPU setup mein CNN ka cost 0.312 milliseconds per image tha, SVM ka 19.748. Yeh batch 128, five repeats ka median hai, input conversion aur features included hain. SVM refit ne saari saved predictions match ki thi, phir timing hui."),
("Nakul", "CNN ke 167,114 trainable values hain. SVM ka count stored decision arrays ka hai, isliye units table mein clear hain. Hybrid ko frozen encoder ke saath LR head bhi ship karna padta hai. File sizes compression aur format par bhi depend karte hain."),
("Nakul", "Integration mein 31 prediction files official labels ke against check hui. IDs, metrics aur hashes match hain. Selected checkpoint ne 12,000 validation predictions reproduce ki. Notebook, report aur slides same reviewed evidence use karte hain."),
("Nakul", "Overall learned features se accuracy aur macro F1 improve hue, aur recorded CPU cost kam hui. Shirt confusion abhi bacha hai. Fusion ka gain support nahi hua. One seed, small grayscale images aur one-machine timing ki wajah se conclusions isi experiment tak bounded hain."),
("Nakul", "Yeh main research sources hain. Full citations report aur literature notes mein hain. Dataset, CNN design, dropout aur AdamW ke ideas yahan se aaye. Thank you, ab questions le sakte hain."),
("All five members", "Offline demo mein notebook kholkar original test ID 40 dekh sakte hain. True class Shirt hai, CNN correct hai aur SVM T-shirt bolta hai. Phir prediction CSV se metric trace karenge. Viva mein split, checkpoint rule aur apne result ka limitation explain karna hai.")
]
(folder / "presenter_notes_hinglish.json").write_text(json.dumps([{"presenter":p,"notes":n} for p,n in notes],indent=2)+"\n")
print("Prepared slide evidence and 20 concise Hinglish notes.")

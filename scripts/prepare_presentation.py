"""Assemble the presentation's inspectable data from frozen evidence. No training."""
import csv
from PIL import Image
from clothing_ml.data import load_official
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read_json(path):
    return json.loads((ROOT/path).read_text())
def csv_rows(path):
    with (ROOT/path).open() as f:
        return list(csv.DictReader(f))
summary=csv_rows('results/full/test_summary.csv')
validation=csv_rows('results/full/validation_summary.csv')
freeze=read_json('results/full/freeze.json')
metrics=read_json('results/full/'+freeze['selected_run_id']+'_test_metrics.json')
complexity=read_json('results/full/complexity.json')
names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
models=['lr','knn','rf','svm']
model_names={'lr':'Logistic Regression','knn':'kNN','rf':'Random Forest','svm':'RBF SVM'}
by={(r['feature'],r['model']):r for r in summary}
validation_by={r['run_id']:r for r in validation}
focus=[0,2,4,6]
confusion=[['Actual / predicted','T-shirt/top','Pullover','Coat','Shirt','Other']]
for i in focus:
    confusion.append([names[i]]+[str(metrics['confusion'][i][j]) for j in focus]+[str(sum(metrics['confusion'][i][j] for j in range(10) if j not in focus))])
results=[['Pipeline','Val. F1','Test acc.','Test F1']]
parameters=[['Pipeline','Selected setting','Scaling']]
complexity_rows=[['Pipeline','Model-specific quantity','MiB']]
complex_by={r['run_id']:r for r in complexity}
for r in summary:
    label=('Pixels' if r['feature']=='pixels' else 'HOG')+' + '+model_names[r['model']]
    results.append([label,f"{float(validation_by[r['run_id']]['macro_f1']):.4f}",f"{float(r['accuracy']):.2%}",f"{float(r['macro_f1']):.4f}"])
for f in freeze['finalists']:
    label=('Pixels' if f['feature']=='pixels' else 'HOG')+' + '+model_names[f['model']]
    param=f['parameters']
    setting={'lr':lambda:f"C = {param['C']:g}",'knn':lambda:f"k = {param['n_neighbors']}",'rf':lambda:'depth = '+('None' if param['max_depth'] is None else str(param['max_depth'])),'svm':lambda:f"C = {param['C']:g}"}[f['model']]()
    scale='StandardScaler' if f['scaler']=='StandardScaler' else 'None'
    parameters.append([label,setting,scale])
    c=complex_by[f['run_id']]
    prefix={'lr':'weights + bias','knn':'stored feature scalars','rf':'tree nodes','svm':'decision-array scalars'}[f['model']]
    complexity_rows.append([label,f"{c['complexity_value']:,} {prefix}",f"{f['serialized_bytes']/2**20:.2f}"])
errors=csv_rows('results/full/error_example_ids.csv')
svm_groups={r['group']:int(r['group_count']) for r in errors if r['model']=='svm'}
data={
 'selected_run_id':freeze['selected_run_id'],'model_names':[model_names[m] for m in models],
 'f1':{f:[float(by[f,m]['macro_f1']) for m in models] for f in ['pixels','hog']},
 'delta':[100*(float(by['hog',m]['macro_f1'])-float(by['pixels',m]['macro_f1'])) for m in models],
 'class_names':names,'recall':metrics['recall'],'confusion_table':confusion,
 'results_table':results,'parameters_table':parameters,'complexity_table':complexity_rows,
 'timing_table':[['Classifier','Pixels (ms/image)','HOG (ms/image)']]+[[model_names[m],f"{float(by['pixels',m]['end_to_end_ms_per_image']):.4f}",f"{float(by['hog',m]['end_to_end_ms_per_image']):.4f}"] for m in models],
 'svm_groups':svm_groups,
 'sources':['results/full/test_summary.csv','results/full/validation_summary.csv','results/full/freeze.json','results/full/complexity.json','results/full/full_hog_svm_3fd9d87a_test_metrics.json','results/full/error_example_ids.csv'],
}
assets=ROOT/'presentation/assets'
assets.mkdir(parents=True,exist_ok=True)
images,labels=load_official('train')
gallery=read_json('figures/eda/gallery_ids.json')
examples=[]
for cls in range(10):
    row=next(r for r in gallery if r['label']==cls)
    Image.fromarray(images[row['sample_id']]).save(assets/f'train_class_{cls}.png')
    examples.append(row)
images,labels=load_official('test')
example_errors=[]
for group in ['Only HOG correct','Both wrong']:
    group_rows=[r for r in errors if r['model']=='svm' and r['group']==group][:2]
    for row in group_rows:
        row={k:(int(v) if k in ['group_count','sample_id','y_true','pixel_prediction','hog_prediction'] else v) for k,v in row.items()}
        Image.fromarray(images[row['sample_id']]).save(assets/f"test_{row['sample_id']}.png")
        example_errors.append(row)
data['gallery_examples']=examples
data['error_examples']=example_errors
(ROOT/'presentation/evidence.json').write_text(json.dumps(data,indent=2)+'\n')
print('Assembled measured presentation evidence, including all eight finalist rows.')

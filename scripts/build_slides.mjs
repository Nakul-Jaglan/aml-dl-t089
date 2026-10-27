// Run in the Artifact Session JavaScript tool. The live deck is the authoring source.
// Presentation evidence is assembled by prepare_presentation.py without training.
var slideFs = await import('node:fs/promises');
var projectDir = '/Users/primemeridiansystems/Desktop/aml-dl-t089';
var deckEvidence = JSON.parse(await slideFs.readFile(projectDir+'/presentation/evidence.json','utf8'));
var speakerRecord = [];
var deckResult = await artifactSession.run(async ({presentation,session})=>{
  if(presentation.slides.items.length) presentation.slides.keep([]);
  presentation.theme.defaultFont = 'Arial';
  var C={ink:'#142F38',muted:'#52656D',paper:'#F7F6F2',teal:'#087F82',blue:'#285F99',orange:'#B86C16',grid:'#D9DFDF'};
  function text(s,value,x,y,w,h,size=27,bold=false,color=C.ink){
    var t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0},text:{text:value,style:{fontSize:size,bold,color,typeface:'Arial',verticalAlignment:'top'}}});
    return t;
  }
  function page(title,presenter,notes,sources=[],appendix=false){
    var s=presentation.slides.add({width:1280,height:720,background:{fill:C.paper}});
    text(s,title,64,42,1150,title.includes('\n')?170:92,title.includes('\n')?58:44,true);
    text(s,(appendix?'Backup  /  ':'')+String(presentation.slides.items.length).padStart(2,'0'),1050,677,170,25,16,false,C.muted);
    var full='Presenter: '+presenter+'\n\n'+notes+'\n\nSources:\n'+sources.join('\n');
    s.speakerNotes.text=full;
    speakerRecord.push({slide:presentation.slides.items.length,title,presenter,notes,sources,appendix});
    return s;
  }
  function para(s,head,body,x,y,w=520){
    text(s,head,x,y,w,45,30,true,C.teal);
    text(s,body,x,y+57,w,155,26,false);
  }
  function table(s,values,x,y,w,h,widths,font=24){
    var t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
    t.borders.assign({fill:C.grid,width:0.7,style:'solid'});
    t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:C.paper,textStyle:{fontSize:font,typeface:'Arial',color:C.ink},margins:{left:10,right:10,top:9,bottom:6},anchor:'center'});
    for(var col=0;col<values[0].length;col++){
      var cell=t.getCell(0,col);cell.fill=C.ink;cell.text.style={fontSize:font,typeface:'Arial',bold:true,color:'#FFFFFF'};
    }
    return t;
  }
  function chart(s,categories,series,x,y,w,h,max=1,format='0.000',horizontal=false){
    return s.charts.add('bar',{
      position:{left:x,top:y,width:w,height:h},categories,series,hasLegend:series.length>1,
      barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:horizontal?50:90},
      legend:{position:'bottom',overlay:false,textStyle:{fontSize:23,fill:C.ink}},
      xAxis:{textStyle:{fontSize:21,fill:C.ink},line:{fill:C.grid,width:1},numberFormatCode:horizontal?format:undefined},
      yAxis:{title:horizontal?undefined:{text:'Macro F1',textStyle:{fontSize:22,fill:C.ink}},min:0,max,numberFormatCode:format,textStyle:{fontSize:20,fill:C.muted},majorGridlines:{fill:C.grid,width:0.7},line:{fill:'none',width:0}},
      dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:21,bold:true,fill:C.ink}},
      chartFill:C.paper,plotAreaFill:C.paper,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
    });
  }
  var s;
  s=page('Automatic clothing\ncategory recognition','Nakul',
    'This project studies automatic category tagging for a small clothing catalogue. The input is one 28 by 28 grayscale image and the output is one of ten Fashion-MNIST categories. This presentation covers the completed classical machine learning experiments in Phase 1. The broader project title includes a small CNN, but its training and comparison belong to Phase 2. Introduce all five team members. The assigned roles are responsibility areas and speaking assignments. They do not replace each member’s review of the shared project.',
    ['docs/course_instructions.md','data/provenance.json']);
  // Cover keeps only the title, context and names.
  text(s,'Phase 1  /  Project 11  /  Section A',68,286,1120,40,25,false,C.teal);
  text(s,'Inventory image tagging with pixels and HOG',68,365,1100,64,33,false);
  text(s,'Nakul    Sudeshhna    Puneet    Samarth    Neelanshu',68,574,1120,42,25,true);

  s=page('Research question and prior work','Nakul',
    'The comparison asks whether fixed local gradient features help more than raw pixel intensities, and whether the answer depends on the classifier. Fashion-MNIST provides a standard dataset with visually similar clothing classes. Dalal and Triggs motivate local gradient histograms, although their original task was human detection. Random Forests and support-vector networks provide different decision rules for the same inputs. This project applies established methods. It does not propose a new algorithm. The CNN is a later test of learned visual features, and no Phase 2 result is available.',
    ['https://arxiv.org/abs/1708.07747','https://doi.org/10.1109/CVPR.2005.177','https://doi.org/10.1023/A:1010933404324','https://doi.org/10.1007/BF00994018','docs/literature_review.md']);
  text(s,'Does local edge structure improve clothing classification?',68,158,1120,100,40,true);
  para(s,'Raw pixels','Intensity at each fixed image position. A simple, direct baseline.',68,300);
  para(s,'HOG descriptors','Histograms of local edge directions. The representation stays fixed during classifier training.',678,300);
  text(s,'Fashion-MNIST: Xiao et al. (2017)     HOG: Dalal & Triggs (2005)',68,588,1130,43,23,false,C.muted);

  s=page('Team responsibilities','Nakul',
    'Introduce the division of responsibility and the speaking handoffs. Nakul covers integration, research and reporting. Sudeshhna explains the dataset and validation split. Puneet explains the feature representations and the simpler classifiers. Samarth explains Random Forest, SVM and model selection. Neelanshu explains metrics, errors and runtime. Each member should rehearse their own slides and understand the entire pipeline. The contribution log is the place to record actual individual reviews or changes.',
    ['docs/contributions.md']);
  table(s,[['Member','Responsibility'],['Nakul','Integration, literature and reporting'],['Sudeshhna','Data quality, EDA and shared splits'],['Puneet','Pixels, HOG, Logistic Regression and kNN'],['Samarth','Random Forest, SVM and experiment management'],['Neelanshu','Evaluation, timing and error analysis']],68,158,1144,430,[250,894],27);

  s=page('Fashion-MNIST dataset','Sudeshhna',
    'Fashion-MNIST has 60,000 official training images and 10,000 official test images. Each image contains 28 by 28 grayscale values, with an intact integer category label. The ten official categories include T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot. The gallery shows seeded examples from the fit pool only. Each training category contains 6,000 images. The benchmark images are small and standardized, so this experiment does not establish performance on phone photographs, cluttered backgrounds or every clothing type.',
    ['https://github.com/zalandoresearch/fashion-mnist','data/training_audit.json','figures/eda/gallery_ids.json']);
  text(s,'60,000 training images     10,000 test images     28 × 28 grayscale',68,145,1140,45,28,true);
  for(var i=0;i<10;i++){
    var x=78+(i%5)*230, y=215+Math.floor(i/5)*210;
    s.images.add({path:projectDir+'/presentation/assets/train_class_'+i+'.png',alt:deckEvidence.class_names[i]+' original training image, ID '+deckEvidence.gallery_examples[i].sample_id,fit:'contain',position:{left:x,top:y,width:150,height:150}});
    text(s,deckEvidence.class_names[i],x,y+159,217,36,22,true);
  }
  text(s,'Official Zalando Fashion-MNIST. Seeded examples from the training fit pool.',68,647,1120,26,19,false,C.muted);

  s=page('Data checks and fair partitions','Sudeshhna',
    'The downloaded IDX files match the published checksums. The audit confirms the expected shape, integer pixel range zero to 255, finite values and all ten balanced labels. It found no all-zero training images, exact training duplicate groups, conflicting-label exact duplicates or exact duplicate overlap between fit and validation. These checks do not verify every semantic label or detect every near-duplicate. A stratified seed-42 split creates 48,000 fit images and 12,000 validation images from the official training set. The official test partition stays separate. Scalers learn their statistics only from fit data, and sample IDs preserve alignment across all pipelines.',
    ['data/training_audit.json','data/provenance.json','data/split_indices.npz','src/clothing_ml/data.py']);
  table(s,[['Partition','Images','Use'],['Fit pool','48,000','Fit all eight final models'],['Validation','12,000','Choose settings and overall model'],['Official test','10,000','Evaluate after selection freeze']],68,160,1144,274,[280,210,654],27);
  text(s,'Seed 42, stratified membership, saved original IDs',68,470,1120,42,29,true,C.teal);
  text(s,'Checks passed: shape, range, labels, exact duplicates and split overlap.\nNear-duplicate detection and a full semantic label audit remain outside scope.',68,535,1120,85,25,false);

  s=page('Pixels and HOG features','Puneet',
    'First divide uint8 image intensities by 255 exactly once. Flattening preserves the 784 numeric values, but the classifier does not receive an explicit neighborhood representation. HOG calculates gradient direction histograms over four by four pixel cells. A 28-pixel side contains seven cells. Two by two cell blocks overlap with a one-cell stride, so there are six by six blocks. Each block has four cells and nine orientation bins, giving 6 times 6 times 4 times 9 equals 1,296 features. L2-Hys normalizes each block. This HOG vector is larger than the pixel vector. It is a fixed descriptor, not a learned CNN representation. Image orientation stays unchanged.',
    ['src/clothing_ml/features.py','figures/eda/hog_example.png','https://doi.org/10.1109/CVPR.2005.177']);
  s.images.add({path:projectDir+'/figures/eda/hog_example.png',alt:'Training Shirt image and its fixed HOG gradient visualization',fit:'contain',position:{left:68,top:170,width:610,height:310}});
  text(s,'Pixels: 28 × 28 = 784\n\nHOG: 6 × 6 × 2 × 2 × 9\n= 1,296 features',730,175,470,230,31,true);
  text(s,'HOG settings: 9 orientations, 4 × 4 cells, 2 × 2 blocks, L2-Hys\nNormalize pixels to [0, 1]. Fit any StandardScaler on the fit pool only.',68,545,1130,90,26,false);

  s=page('Logistic Regression and kNN','Puneet',
    'Multiclass Logistic Regression learns one linear score per category and fits the scores with regularization. A larger C allows weaker regularization. It is fast to evaluate, but a linear boundary can miss complex relationships. kNN stores the fit examples and assigns labels using nearby examples in feature space. Here it uses brute-force Euclidean distances and distance-weighted voting. k controls how many neighbors vote. kNN has almost no optimization cost, but prediction can be expensive because it compares a query with stored examples. LR uses StandardScaler for both feature types. HOG kNN uses StandardScaler, while pixel kNN uses normalized intensities without learned standardization. The comparison therefore concerns declared pipelines rather than feature extraction in isolation.',
    ['src/clothing_ml/models.py','docs/protocol.md','results/full/freeze.json']);
  para(s,'Logistic Regression','Learns linear category scores. C controls regularization strength.',68,170);
  para(s,'k nearest neighbors','Uses distance-weighted votes from stored examples. k controls the neighborhood.',678,170);
  table(s,[['Pipeline','Preprocessing'],['LR with pixels or HOG','Training-only StandardScaler'],['kNN with pixels','Pixel normalization only'],['kNN with HOG','Training-only StandardScaler']],68,440,1144,196,[480,664],25);

  s=page('Random Forest and RBF SVM','Samarth',
    'Random Forest averages decisions from randomized trees. Bootstrap samples and random feature subsets reduce dependence between the trees. Each forest here contains 200 trees and uses square-root feature sampling. Maximum depth controls how detailed the tree rules can become. RF receives no StandardScaler because its threshold rules do not require it. RBF SVM uses a nonlinear similarity function and a margin-based decision rule. C controls the penalty for training errors relative to margin regularization, and gamma controls how local the RBF similarity is. Both SVM representations use StandardScaler. Gamma stays at scale, C is selected from a small grid, and probability estimation is disabled. These models illustrate different ways to use the same fixed image features.',
    ['src/clothing_ml/models.py','results/full/freeze.json','https://doi.org/10.1023/A:1010933404324','https://doi.org/10.1007/BF00994018']);
  para(s,'Random Forest','200 randomized decision trees. Depth controls rule complexity. No StandardScaler.',68,175);
  para(s,'RBF SVM','Nonlinear similarity and margin decisions. StandardScaler, gamma = scale, no probability estimation.',678,175);
  text(s,'The same fit and validation IDs support all four model families',68,492,1120,100,36,true,C.teal);

  s=page('Model selection before test access','Samarth',
    'The smoke profile checks that all pipeline types run. It cannot unlock official-test evaluation. The compact search uses a stratified 12,000-image subset of the fit pool and the same 12,000-image validation set. Each of eight model-feature rows has two candidates, giving 16 search fits. LR searches C 0.1 and 1. kNN searches k 3 and 5. RF searches maximum depth 20 and None. RBF SVM searches C 1 and 10. Selected settings then train afresh on all 48,000 fit images. Full-profile validation macro F1 chooses the overall pipeline. Accuracy, prediction time and run ID break ties in that order. A hashed manifest freezes artifacts and selection before official-test access. This is a compact search, not exhaustive optimization.',
    ['docs/protocol.md','results/budget/validation_candidates.csv','results/full/freeze.json','results/full/test_receipt.json']);
  table(s,[['Stage','Images used','Decision'],['Smoke check','2,000','Verify the pipeline'],['Compact development search','12,000','Two candidates per feature/model row'],['Final fitting','48,000','Fit eight selected configurations'],['Freeze, then official test','10,000 test','Evaluate the fixed finalists']],68,156,1144,350,[440,190,514],25);
  text(s,'Primary selection metric: validation macro F1',68,547,1120,55,34,true,C.teal);
  text(s,'All eight final models share the same 48,000 fit images.',68,610,1120,35,25,false);

  s=page('Evaluation measures','Neelanshu',
    'Accuracy is the fraction of all images that receive the correct category. For a single class, precision measures how often that prediction is correct, and recall measures how many true members of that class the model finds. F1 is two times precision times recall divided by their sum. Macro F1 averages the ten class F1 values with equal class weight. It is not the harmonic mean of macro precision and macro recall. Per-class recall exposes a weak category even when overall accuracy is high. Confusion matrix rows are true categories and columns are predicted categories. We also measure inference time and model-specific complexity because classification quality alone does not determine usability.',
    ['src/clothing_ml/evaluate.py','tests/test_integrity.py','results/full/per_class_recall.csv']);
  para(s,'Accuracy','Correct predictions / all images',68,173);
  para(s,'Macro F1','Mean of the ten class F1 scores. Each category has equal weight.',678,173);
  para(s,'Per-class recall','Correctly identified members of a class / all true members of that class',68,417);
  para(s,'Operational cost','Repeated inference measurements and model-specific storage quantities',678,417);

  s=page('HOG helps three classifier families','Neelanshu',
    'This native chart compares official-test macro F1 across the eight frozen pipelines. The HOG SVM row had already won validation selection. It reaches 90.91 percent test accuracy and macro F1 0.9090. Relative to pixels, HOG increases macro F1 by 3.63 percentage points for LR, 2.26 for kNN and 1.65 for SVM. It reduces RF macro F1 by 0.35 points. The effect therefore depends on the classifier and preprocessing. These results come from one split and a compact search. Do not claim a statistically significant difference, a universal winner or improvement caused only by HOG in isolation.',
    ['results/full/test_summary.csv','results/full/freeze.json','presentation/evidence.json']);
  chart(s,deckEvidence.model_names,[{name:'Pixels',values:deckEvidence.f1.pixels.map(v=>Number(v.toFixed(6))),fill:C.blue},{name:'HOG',values:deckEvidence.f1.hog.map(v=>Number(v.toFixed(6))),fill:C.orange}],68,154,1144,430,1,'0.000');
  text(s,'Selected HOG + RBF SVM: 90.91% accuracy, 0.9090 macro F1',68,606,1144,48,29,true,C.teal);

  s=page('Shirt recall remains the weakest','Neelanshu',
    'The chart shows official-test recall for the selected HOG SVM only. Each category has 1,000 test images. Shirt recall is 73.3 percent, which means 733 of the 1,000 true Shirts receive the Shirt label. Bag recall is 98.3 percent. The difference demonstrates why a single aggregate accuracy value is insufficient. Some upper-body categories share outlines and local edge patterns at 28 by 28 resolution. That is a plausible interpretation of the observed errors, not a proven causal mechanism. The next slide examines measured confusion counts rather than relying on visual impressions alone.',
    ['results/full/full_hog_svm_3fd9d87a_test_metrics.json','results/full/per_class_recall.csv']);
  chart(s,deckEvidence.class_names,[{name:'HOG + RBF SVM recall',values:deckEvidence.recall,fill:C.teal}],60,144,1150,455,1.08,'0.0%',true);
  text(s,'Selected model. Official test: 1,000 images per category.',68,625,1120,31,23,false,C.muted);

  s=page('Confusion among upper-body categories','Neelanshu',
    'Read the table by row. The actual category is on the left and each column gives its predicted category. The Other column combines the six categories not displayed separately, so every row still sums to 1,000. Of the true T-shirt/top images, 104 are predicted as Shirt. Of the true Shirts, 103 are predicted as T-shirt/top, 63 as Pullover and 64 as Coat. This mutual confusion explains much of the weakest recall. The diagonal counts are correct classifications. The table is a focused view of the full ten by ten confusion matrix, not a separate experiment or a newly selected subset for model tuning.',
    ['results/full/full_hog_svm_3fd9d87a_test_metrics.json','presentation/evidence.json']);
  table(s,deckEvidence.confusion_table,68,165,1144,338,[300,178,166,166,166,168],26);
  text(s,'Rows: true labels. Columns: predictions. Each row totals 1,000.',68,545,1135,42,26,true,C.teal);
  text(s,'Other combines the six remaining predicted categories.',68,603,1135,35,24,false,C.muted);

  s=page('Paired SVM prediction examples','Neelanshu',
    'These original test images compare the frozen pixel SVM and HOG SVM on identical sample IDs. The examples follow a deterministic first-ID rule within each outcome group. They do not estimate group prevalence. In the HOG-only-correct group, HOG succeeds where the pixel SVM fails. In the both-wrong group, fixed gradients still fail. Read the true label and both predictions before interpreting a shape. A single visual example can suggest a reason to investigate, but it does not establish a causal explanation or prove a dataset label is wrong. Group counts for the complete official test are in the saved error evidence.',
    ['results/full/error_example_ids.csv','report/figures/error_examples.png','src/clothing_ml/plots.py']);
  for(var i=0;i<4;i++){
    var e=deckEvidence.error_examples[i],x=68+i*290;
    text(s,(i<2?'Only HOG correct':'Both wrong')+'\nTest ID '+e.sample_id,x,163,272,74,25,true,C.teal);
    s.images.add({path:projectDir+'/presentation/assets/test_'+e.sample_id+'.png',alt:'Original test image '+e.sample_id+', true '+deckEvidence.class_names[e.y_true],fit:'contain',position:{left:x,top:264,width:220,height:220}});
    text(s,'True: '+deckEvidence.class_names[e.y_true]+'\nPixels: '+deckEvidence.class_names[e.pixel_prediction]+'\nHOG: '+deckEvidence.class_names[e.hog_prediction],x,501,273,99,23,false);
  }
  text(s,'First sample IDs per outcome group. Examples illustrate errors, not prevalence.',68,635,1140,32,23,false,C.muted);

  s=page('Inference cost changes the trade-off','Neelanshu',
    'The table reports median end-to-end inference time per image, amortized over batches. It includes normalization, feature extraction, training-fitted scaling where applicable and classifier prediction. It excludes disk reads and model loading. Measurements use official-test IDs zero through 1,023, batches of 128, one warm-up batch and five repeats, with one native thread on the recorded ARM64 macOS machine. HOG SVM is the strongest measured classifier but costs 21.7090 milliseconds per image in this setting. Pixel LR costs 0.0014 milliseconds, with lower accuracy. These are batch costs on one machine, not interactive single-image latency or guaranteed deployment performance. Pipeline-only timing and repeat ranges remain in the evidence files.',
    ['results/full/timing.json','results/full/test_summary.csv']);
  table(s,deckEvidence.timing_table,68,168,1144,324,[480,332,332],28);
  text(s,'Median end-to-end cost, batch 128, 5 repeats, 1 native thread',68,533,1144,42,28,true,C.teal);
  text(s,'Includes feature extraction. Excludes loading. One machine and 1,024 fixed images.',68,600,1144,40,23,false,C.muted);

  s=page('Reproducible Phase 1 evidence','Nakul',
    'The repository retains source modules, exact saved split membership, dataset checksums, candidate logs and the hashed selection manifest. Compressed predictions preserve original IDs and true labels. An independent audit rebuilds the reported metrics from the prediction files, while the executed notebook presents the analysis with saved outputs. Eight integrity tests passed, including split checks, metric checks, scaling behavior, prediction alignment and the test gate. The GitHub workflow repeats the lightweight checks. A fresh reproduction writes to an isolated result directory so that the checked-in frozen evidence stays intact. Large model bundles and downloaded data stay local. The deck itself reads recorded results and does not retrain any model.',
    ['results/full/evidence_audit.json','tests/test_integrity.py','.github/workflows/checks.yml','notebooks/00_phase1_master.ipynb','docs/protocol.md']);
  para(s,'Trace each result','Saved IDs, candidate logs, prediction files and a hashed freeze manifest',68,165);
  para(s,'Check the calculations','Executed notebook, independent metric audit and eight integrity tests',678,165);
  text(s,'github.com/Nakul-Jaglan/aml-dl-t089',68,461,1140,53,35,true,C.teal);
  text(s,'Source, results, figures and report share the same frozen experiment evidence.',68,549,1140,78,28,false);

  s=page('Conclusions and Phase 2 proposal','Nakul',
    'Phase 1 establishes a controlled comparison of two feature representations and four classical classifiers. The validation-selected HOG SVM achieves 90.91 percent test accuracy, and HOG helps three model families under this protocol. Shirt remains the weakest class and inference cost increases substantially for the strongest model. Limits include one split, a small search, model-specific scaling and standardized low-resolution images. Phase 2 should test a small two- or three-block CNN with pooling and dropout, training-only validation for early stopping and architecture choices, and the same evaluation measures. The learned hierarchy may help distinguish similar clothes, but that remains an unmeasured hypothesis. Test results already seen in Phase 1 must not guide CNN tuning.',
    ['results/full/test_summary.csv','results/full/full_hog_svm_3fd9d87a_test_metrics.json','docs/course_instructions.md','report/main.tex']);
  text(s,'90.91%',68,160,560,110,85,true,C.teal);
  text(s,'Test accuracy of HOG + RBF SVM',68,285,1144,53,32,true);
  text(s,'Shirt recall: 73.3%. Higher accuracy carries a larger inference cost.',68,370,1130,84,29,false);
  text(s,'Limits: one split, compact search, fixed preprocessing and 28 × 28 images.',68,452,1130,40,24,false,C.muted);
  text(s,'Phase 2: test a small CNN on the same saved partitions.\nCNN performance is unmeasured.',68,512,1130,105,32,true);

  s=page('Selected references','Nakul',
    'The full literature review and bibliography are available in the report and repository. Fashion-MNIST supplies the benchmark. Dalal and Triggs motivate HOG. Breiman introduces Random Forests. Cortes and Vapnik supply support-vector-network background. Scikit-learn provides the implementation and evaluation tools. LeCun and colleagues provide background for the later CNN. The team literature notes distinguish full-text reading from abstract-only access. The original HOG paper studies human detection, so cite it as the source of the descriptor rather than direct evidence about this clothing experiment.',
    ['docs/literature_review.md','report/references.bib']);
  text(s,'Xiao et al. (2017)  Fashion-MNIST\narxiv.org/abs/1708.07747\n\nDalal & Triggs (2005)  Histograms of Oriented Gradients\ndoi.org/10.1109/CVPR.2005.177\n\nBreiman (2001)  Random Forests\ndoi.org/10.1023/A:1010933404324',68,157,1135,390,26,false);
  text(s,'Cortes & Vapnik (1995), Support-vector networks\nPedregosa et al. (2011), Scikit-learn: Machine Learning in Python\nLeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition',68,555,1135,103,23,false,C.muted);

  s=page('All eight finalists','Samarth and Neelanshu',
    'This backup table shows validation macro F1 alongside official-test accuracy and macro F1 for every finalist. All final rows use 48,000 fit examples and the same 12,000 validation IDs. HOG SVM has the highest validation macro F1, 0.9114, which selected it before test access. Its test F1 is 0.9090. Compare like-for-like model families when explaining the feature effect. These values are rounded for display, while the CSVs retain full precision. The selection manifest records exact artifacts and parameters.',
    ['results/full/validation_summary.csv','results/full/test_summary.csv'],true);
  table(s,deckEvidence.results_table,68,149,1144,468,[515,205,215,209],24);
  text(s,'Selection used validation F1. The official test did not choose the winner.',68,637,1110,27,22,false,C.muted);

  s=page('Selected settings and search scope','Samarth',
    'The table lists the settings selected separately for each feature-model pair, plus the scaling policy. The compact search examines LR C 0.1 and 1, kNN k 3 and 5, RF depth 20 and None, and SVM C 1 and 10. RF keeps 200 trees and square-root feature sampling. LR uses LBFGS and at most 2,000 iterations. kNN uses distance weighting and brute-force Euclidean distance. SVM uses RBF, gamma scale and probability false. Other defaults and installed versions are captured in the saved run records. A small grid is a resource-conscious development choice and does not prove that each model family reached its best possible performance.',
    ['results/full/freeze.json','results/budget/validation_candidates.csv','src/clothing_ml/models.py'],true);
  table(s,deckEvidence.parameters_table,68,149,1144,468,[480,300,364],24);
  text(s,'Two candidates per feature/model pair. Fixed RF: 200 trees. Fixed SVM: RBF.',68,635,1130,28,22,false,C.muted);

  s=page('Model size and complexity','Nakul',
    'Classical models need different complexity descriptions. LR has learned coefficient and intercept scalars. kNN stores training feature values and has no comparable learned coefficient count. RF complexity here counts all tree nodes. SVM uses the stored support-vector and decision-array scalar count as a storage proxy. None of these quantities is directly interchangeable with neural-network trainable parameters. The right column reports compressed serialized bundle size in MiB, which is a common artifact measure but is not peak RAM. HOG SVM stores 20,916 support vectors. Scaler mean and scale arrays are counted separately in the evidence. Do not interpret the larger quantity as a general measure of statistical quality.',
    ['results/full/complexity.json','results/full/freeze.json','results/full/test_summary.csv'],true);
  table(s,deckEvidence.complexity_table,50,149,1180,468,[400,650,130],22);
  text(s,'Counts differ by model family. MiB is serialized size, not peak memory.',68,636,1130,29,22,false,C.muted);

  s=page('Viva and saved-prediction demo','All five members',
    'For an offline demo, open the executed master notebook or its HTML companion. Find original test ID zero in the saved predictions. Its true category is Ankle boot and the selected HOG SVM predicts Ankle boot. Show the matching ID in another model’s prediction file to explain alignment. Recompute accuracy as the count of correct predictions divided by 10,000, using the saved CSV rather than training again. For viva practice, each member should explain the fit/validation/test distinction, the HOG dimension calculation and one weakness of the selected model. If asked whether CNN is better, say it has not been evaluated yet. The saved artifacts support the demonstration without network access.',
    ['notebooks/00_phase1_master.ipynb','notebooks/00_phase1_master.html','results/full/predictions/full_hog_svm_3fd9d87a_test.csv.gz','docs/START_HERE_NAKUL.md'],true);
  para(s,'Core questions','Why keep validation separate?\nWhy is HOG 1,296-dimensional?\nWhy can scaling change kNN?',68,170);
  para(s,'Offline demonstration','Open the executed notebook.\nInspect original test ID 0.\nRecompute a saved metric.',678,170);
  text(s,'Explain one error and one limitation in your own words',68,502,1130,105,38,true,C.teal);

  session.setActiveSlide(presentation.slides.items[0]);
  nodeRepl.write({slides:presentation.slides.items.length});
  // Render directly in the live session, then export portable copies for the repository.
  await presentation.render({format:'png',scale:1});
  await presentation.export({format:'pptx',fileName:'Phase_1_Clothing_Category_Recognition.pptx'});

},{artifactType:'presentation',title:'Clothing Category Recognition — Phase 1'});
await slideFs.writeFile(projectDir+'/presentation/speaker_notes.md','# Phase 1 presenter notes\n\nSlides 1–18 form the main presentation. Slides 19–22 are backup slides.\nSpeaking assignments follow the team responsibilities. Adjust pacing to the faculty time limit.\n\n'+speakerRecord.map(r=>'## '+r.slide+'. '+r.title.replaceAll('\n',' ')+'\n\nPresenter: **'+r.presenter+'**'+(r.appendix?' (backup)':'')+'\n\n'+r.notes+'\n\nSources:\n'+r.sources.map(v=>'- '+v).join('\n')).join('\n\n'));
nodeRepl.write(deckResult);

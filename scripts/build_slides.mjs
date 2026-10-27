// Run in the Artifact Session JavaScript tool. The live deck is the authoring source.
// Presentation evidence is assembled by prepare_presentation.py without training.
var slideFs = await import('node:fs/promises');
var projectDir = '/Users/primemeridiansystems/Desktop/aml-dl-t089';
var deckEvidence = JSON.parse(await slideFs.readFile(projectDir+'/presentation/evidence.json','utf8'));
var hinglishNotes = JSON.parse(await slideFs.readFile(projectDir+'/presentation/presenter_notes_hinglish.json','utf8'));
if(hinglishNotes.length!==22) throw new Error('Expected notes for all 22 slides');
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
    hinglishNotes[0],
    ['docs/course_instructions.md','data/provenance.json']);
  // Cover keeps only the title, context and names.
  text(s,'Phase 1  /  Project 11  /  Section A',68,286,1120,40,25,false,C.teal);
  text(s,'Inventory image tagging with pixels and HOG',68,365,1100,64,33,false);
  text(s,'Nakul    Sudeshhna    Puneet    Samarth    Neelanshu',68,574,1120,42,25,true);

  s=page('Research question and prior work','Nakul',
    hinglishNotes[1],
    ['https://arxiv.org/abs/1708.07747','https://doi.org/10.1109/CVPR.2005.177','https://doi.org/10.1023/A:1010933404324','https://doi.org/10.1007/BF00994018','docs/literature_review.md']);
  text(s,'Does local edge structure improve clothing classification?',68,158,1120,100,40,true);
  para(s,'Raw pixels','Intensity at each fixed image position. A simple, direct baseline.',68,300);
  para(s,'HOG descriptors','Histograms of local edge directions. The representation stays fixed during classifier training.',678,300);
  text(s,'Fashion-MNIST: Xiao et al. (2017)     HOG: Dalal & Triggs (2005)',68,588,1130,43,23,false,C.muted);

  s=page('Team responsibilities','Nakul',
    hinglishNotes[2],
    ['docs/contributions.md']);
  table(s,[['Member','Responsibility'],['Nakul','Integration, literature and reporting'],['Sudeshhna','Data quality, EDA and shared splits'],['Puneet','Pixels, HOG, Logistic Regression and kNN'],['Samarth','Random Forest, SVM and experiment management'],['Neelanshu','Evaluation, timing and error analysis']],68,158,1144,430,[250,894],27);

  s=page('Fashion-MNIST dataset','Sudeshhna',
    hinglishNotes[3],
    ['https://github.com/zalandoresearch/fashion-mnist','data/training_audit.json','figures/eda/gallery_ids.json']);
  text(s,'60,000 training images     10,000 test images     28 × 28 grayscale',68,145,1140,45,28,true);
  for(var i=0;i<10;i++){
    var x=78+(i%5)*230, y=215+Math.floor(i/5)*210;
    s.images.add({path:projectDir+'/presentation/assets/train_class_'+i+'.png',alt:deckEvidence.class_names[i]+' original training image, ID '+deckEvidence.gallery_examples[i].sample_id,fit:'contain',position:{left:x,top:y,width:150,height:150}});
    text(s,deckEvidence.class_names[i],x,y+159,217,36,22,true);
  }
  text(s,'Official Zalando Fashion-MNIST. Seeded examples from the training fit pool.',68,647,1120,26,19,false,C.muted);

  s=page('Data checks and fair partitions','Sudeshhna',
    hinglishNotes[4],
    ['data/training_audit.json','data/provenance.json','data/split_indices.npz','src/clothing_ml/data.py']);
  table(s,[['Partition','Images','Use'],['Fit pool','48,000','Fit all eight final models'],['Validation','12,000','Choose settings and overall model'],['Official test','10,000','Evaluate after selection freeze']],68,160,1144,274,[280,210,654],27);
  text(s,'Seed 42, stratified membership, saved original IDs',68,470,1120,42,29,true,C.teal);
  text(s,'Checks passed: shape, range, labels, exact duplicates and split overlap.\nNear-duplicate detection and a full semantic label audit remain outside scope.',68,535,1120,85,25,false);

  s=page('Pixels and HOG features','Puneet',
    hinglishNotes[5],
    ['src/clothing_ml/features.py','figures/eda/hog_example.png','https://doi.org/10.1109/CVPR.2005.177']);
  s.images.add({path:projectDir+'/figures/eda/hog_example.png',alt:'Training Shirt image and its fixed HOG gradient visualization',fit:'contain',position:{left:68,top:170,width:610,height:310}});
  text(s,'Pixels: 28 × 28 = 784\n\nHOG: 6 × 6 × 2 × 2 × 9\n= 1,296 features',730,175,470,230,31,true);
  text(s,'HOG settings: 9 orientations, 4 × 4 cells, 2 × 2 blocks, L2-Hys\nNormalize pixels to [0, 1]. Fit any StandardScaler on the fit pool only.',68,545,1130,90,26,false);

  s=page('Logistic Regression and kNN','Puneet',
    hinglishNotes[6],
    ['src/clothing_ml/models.py','docs/protocol.md','results/full/freeze.json']);
  para(s,'Logistic Regression','Learns linear category scores. C controls regularization strength.',68,170);
  para(s,'k nearest neighbors','Uses distance-weighted votes from stored examples. k controls the neighborhood.',678,170);
  table(s,[['Pipeline','Preprocessing'],['LR with pixels or HOG','Training-only StandardScaler'],['kNN with pixels','Pixel normalization only'],['kNN with HOG','Training-only StandardScaler']],68,440,1144,196,[480,664],25);

  s=page('Random Forest and RBF SVM','Samarth',
    hinglishNotes[7],
    ['src/clothing_ml/models.py','results/full/freeze.json','https://doi.org/10.1023/A:1010933404324','https://doi.org/10.1007/BF00994018']);
  para(s,'Random Forest','200 randomized decision trees. Depth controls rule complexity. No StandardScaler.',68,175);
  para(s,'RBF SVM','Nonlinear similarity and margin decisions. StandardScaler, gamma = scale, no probability estimation.',678,175);
  text(s,'The same fit and validation IDs support all four model families',68,492,1120,100,36,true,C.teal);

  s=page('Model selection before test access','Samarth',
    hinglishNotes[8],
    ['docs/protocol.md','results/budget/validation_candidates.csv','results/full/freeze.json','results/full/test_receipt.json']);
  table(s,[['Stage','Images used','Decision'],['Smoke check','2,000','Verify the pipeline'],['Compact development search','12,000','Two candidates per feature/model row'],['Final fitting','48,000','Fit eight selected configurations'],['Freeze, then official test','10,000 test','Evaluate the fixed finalists']],68,156,1144,350,[440,190,514],25);
  text(s,'Primary selection metric: validation macro F1',68,547,1120,55,34,true,C.teal);
  text(s,'All eight final models share the same 48,000 fit images.',68,610,1120,35,25,false);

  s=page('Evaluation measures','Neelanshu',
    hinglishNotes[9],
    ['src/clothing_ml/evaluate.py','tests/test_integrity.py','results/full/per_class_recall.csv']);
  para(s,'Accuracy','Correct predictions / all images',68,173);
  para(s,'Macro F1','Mean of the ten class F1 scores. Each category has equal weight.',678,173);
  para(s,'Per-class recall','Correctly identified members of a class / all true members of that class',68,417);
  para(s,'Operational cost','Repeated inference measurements and model-specific storage quantities',678,417);

  s=page('HOG helps three classifier families','Neelanshu',
    hinglishNotes[10],
    ['results/full/test_summary.csv','results/full/freeze.json','presentation/evidence.json']);
  chart(s,deckEvidence.model_names,[{name:'Pixels',values:deckEvidence.f1.pixels.map(v=>Number(v.toFixed(6))),fill:C.blue},{name:'HOG',values:deckEvidence.f1.hog.map(v=>Number(v.toFixed(6))),fill:C.orange}],68,154,1144,430,1,'0.000');
  text(s,'Selected HOG + RBF SVM: 90.91% accuracy, 0.9090 macro F1',68,606,1144,48,29,true,C.teal);

  s=page('Shirt recall remains the weakest','Neelanshu',
    hinglishNotes[11],
    ['results/full/full_hog_svm_3fd9d87a_test_metrics.json','results/full/per_class_recall.csv']);
  chart(s,deckEvidence.class_names,[{name:'HOG + RBF SVM recall',values:deckEvidence.recall,fill:C.teal}],60,144,1150,455,1.08,'0.0%',true);
  text(s,'Selected model. Official test: 1,000 images per category.',68,625,1120,31,23,false,C.muted);

  s=page('Confusion among upper-body categories','Neelanshu',
    hinglishNotes[12],
    ['results/full/full_hog_svm_3fd9d87a_test_metrics.json','presentation/evidence.json']);
  table(s,deckEvidence.confusion_table,68,165,1144,338,[300,178,166,166,166,168],26);
  text(s,'Rows: true labels. Columns: predictions. Each row totals 1,000.',68,545,1135,42,26,true,C.teal);
  text(s,'Other combines the six remaining predicted categories.',68,603,1135,35,24,false,C.muted);

  s=page('Paired SVM prediction examples','Neelanshu',
    hinglishNotes[13],
    ['results/full/error_example_ids.csv','report/figures/error_examples.png','src/clothing_ml/plots.py']);
  for(var i=0;i<4;i++){
    var e=deckEvidence.error_examples[i],x=68+i*290;
    text(s,(i<2?'Only HOG correct':'Both wrong')+'\nTest ID '+e.sample_id,x,163,272,74,25,true,C.teal);
    s.images.add({path:projectDir+'/presentation/assets/test_'+e.sample_id+'.png',alt:'Original test image '+e.sample_id+', true '+deckEvidence.class_names[e.y_true],fit:'contain',position:{left:x,top:264,width:220,height:220}});
    text(s,'True: '+deckEvidence.class_names[e.y_true]+'\nPixels: '+deckEvidence.class_names[e.pixel_prediction]+'\nHOG: '+deckEvidence.class_names[e.hog_prediction],x,501,273,99,23,false);
  }
  text(s,'First sample IDs per outcome group. Examples illustrate errors, not prevalence.',68,635,1140,32,23,false,C.muted);

  s=page('Inference cost changes the trade-off','Neelanshu',
    hinglishNotes[14],
    ['results/full/timing.json','results/full/test_summary.csv']);
  table(s,deckEvidence.timing_table,68,168,1144,324,[480,332,332],28);
  text(s,'Median end-to-end cost, batch 128, 5 repeats, 1 native thread',68,533,1144,42,28,true,C.teal);
  text(s,'Includes feature extraction. Excludes loading. One machine and 1,024 fixed images.',68,600,1144,40,23,false,C.muted);

  s=page('Reproducible Phase 1 evidence','Nakul',
    hinglishNotes[15],
    ['results/full/evidence_audit.json','tests/test_integrity.py','.github/workflows/checks.yml','notebooks/00_phase1_master.ipynb','docs/protocol.md']);
  para(s,'Trace each result','Saved IDs, candidate logs, prediction files and a hashed freeze manifest',68,165);
  para(s,'Check the calculations','Executed notebook, independent metric audit and eight integrity tests',678,165);
  text(s,'github.com/Nakul-Jaglan/aml-dl-t089',68,461,1140,53,35,true,C.teal);
  text(s,'Source, results, figures and report share the same frozen experiment evidence.',68,549,1140,78,28,false);

  s=page('Conclusions and Phase 2 proposal','Nakul',
    hinglishNotes[16],
    ['results/full/test_summary.csv','results/full/full_hog_svm_3fd9d87a_test_metrics.json','docs/course_instructions.md','report/main.tex']);
  text(s,'90.91%',68,160,560,110,85,true,C.teal);
  text(s,'Test accuracy of HOG + RBF SVM',68,285,1144,53,32,true);
  text(s,'Shirt recall: 73.3%. Higher accuracy carries a larger inference cost.',68,370,1130,84,29,false);
  text(s,'Limits: one split, compact search, fixed preprocessing and 28 × 28 images.',68,452,1130,40,24,false,C.muted);
  text(s,'Phase 2: test a small CNN on the same saved partitions.\nCNN performance is unmeasured.',68,512,1130,105,32,true);

  s=page('Selected references','Nakul',
    hinglishNotes[17],
    ['docs/literature_review.md','report/references.bib']);
  text(s,'Xiao et al. (2017)  Fashion-MNIST\narxiv.org/abs/1708.07747\n\nDalal & Triggs (2005)  Histograms of Oriented Gradients\ndoi.org/10.1109/CVPR.2005.177\n\nBreiman (2001)  Random Forests\ndoi.org/10.1023/A:1010933404324',68,157,1135,390,26,false);
  text(s,'Cortes & Vapnik (1995), Support-vector networks\nPedregosa et al. (2011), Scikit-learn: Machine Learning in Python\nLeCun et al. (1998), Gradient-Based Learning Applied to Document Recognition',68,555,1135,103,23,false,C.muted);

  s=page('All eight finalists','Samarth and Neelanshu',
    hinglishNotes[18],
    ['results/full/validation_summary.csv','results/full/test_summary.csv'],true);
  table(s,deckEvidence.results_table,68,149,1144,468,[515,205,215,209],24);
  text(s,'Selection used validation F1. The official test did not choose the winner.',68,637,1110,27,22,false,C.muted);

  s=page('Selected settings and search scope','Samarth',
    hinglishNotes[19],
    ['results/full/freeze.json','results/budget/validation_candidates.csv','src/clothing_ml/models.py'],true);
  table(s,deckEvidence.parameters_table,68,149,1144,468,[480,300,364],24);
  text(s,'Two candidates per feature/model pair. Fixed RF: 200 trees. Fixed SVM: RBF.',68,635,1130,28,22,false,C.muted);

  s=page('Model size and complexity','Nakul',
    hinglishNotes[20],
    ['results/full/complexity.json','results/full/freeze.json','results/full/test_summary.csv'],true);
  table(s,deckEvidence.complexity_table,50,149,1180,468,[400,650,130],22);
  text(s,'Counts differ by model family. MiB is serialized size, not peak memory.',68,636,1130,29,22,false,C.muted);

  s=page('Viva and saved-prediction demo','All five members',
    hinglishNotes[21],
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
await slideFs.writeFile(projectDir+'/presentation/speaker_notes.md','# Phase 1 presenter notes\n\nSlides 1–18 form the main presentation. Slides 19–22 are backup slides.\nShort spoken notes use Roman-script Hinglish. Sources are reference material, not spoken lines.\nSpeaking assignments follow the team responsibilities. Adjust pacing to the faculty time limit.\n\n'+speakerRecord.map(r=>'## '+r.slide+'. '+r.title.replaceAll('\n',' ')+'\n\nPresenter: **'+r.presenter+'**'+(r.appendix?' (backup)':'')+'\n\n'+r.notes+'\n\nSources:\n'+r.sources.map(v=>'- '+v).join('\n')).join('\n\n'));
nodeRepl.write(deckResult);

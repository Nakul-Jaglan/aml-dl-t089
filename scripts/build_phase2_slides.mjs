// Portable file authoring for the final Phase 2 deck when Spaces creation is unavailable.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const runtimeRequire = createRequire(process.env.RUNTIME_NODE_MODULES
    ? path.join(process.env.RUNTIME_NODE_MODULES, '..', 'package.json') : import.meta.url);
const {Presentation, PresentationFile} = await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);

const root = process.cwd();
const buildDir = path.join(root, 'work/phase2_integration/deck');
const outputDir = path.join(root, 'presentation/phase2');
await fs.mkdir(buildDir, {recursive:true});
const data = JSON.parse(await fs.readFile(path.join(outputDir, 'evidence.json'), 'utf8'));
const notes = JSON.parse(await fs.readFile(path.join(outputDir, 'presenter_notes_hinglish.json'), 'utf8'));
const presentation = Presentation.create({slideSize:{width:1280,height:720}});
presentation.theme.defaultFont = 'Arial';
const colors = {ink:'#142F38', muted:'#52656D', paper:'#F7F6F2', teal:'#087F82', blue:'#285F99', orange:'#B86C16', grid:'#D9DFDF'};
const sources = [];
const titles = [];
const nativeTables = [];
const nativeCharts = [];

function text(slide, value, left, top, width, height, size=27, bold=false, color=colors.ink) {
    return slide.shapes.add({geometry:'textbox', position:{left,top,width,height}, fill:'none',
        line:{fill:'none',width:0}, text:{text:value,style:{fontSize:size,bold,color,typeface:'Arial',verticalAlignment:'top'}}});
}
function page(title, sourceFiles) {
    const index = presentation.slides.items.length;
    const slide = presentation.slides.add();
    slide.background.fill = colors.paper;
    text(slide,title,64,42,1150,title.includes('\n')?170:92,title.includes('\n')?58:44,true);
    text(slide,String(index+1).padStart(2,'0'),1120,677,95,25,16,false,colors.muted);
    slide.speakerNotes.text = 'Presenter: '+notes[index].presenter+'\n\n'+notes[index].notes+'\n\nSources:\n'+sourceFiles.join('\n');
    sources.push(sourceFiles);
    titles.push(title);
    return slide;
}
function table(slide, values, top=160, height=360, widths=[480,332,332], size=26) {
    nativeTables.push(presentation.slides.items.length);
    const owner = slide.tables.add({rows:values.length,columns:values[0].length,left:68,top,width:1144,height,columnWidths:widths,values});
    owner.borders.assign({fill:colors.grid,width:.7,style:'solid'});
    owner.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({
        fill:colors.paper,textStyle:{fontSize:size,typeface:'Arial',color:colors.ink},
        margins:{left:10,right:10,top:9,bottom:6},anchor:'center'});
    for (let column=0;column<values[0].length;column++) {
        owner.getCell(0,column).fill=colors.ink;
        owner.getCell(0,column).text.style={fontSize:size,typeface:'Arial',bold:true,color:'#FFFFFF'};
    }
    return owner;
}
function bars(slide,categories,series,{left=68,top=155,width=1144,height=435,label='Macro F1',format='0.000',horizontal=false,max=1.04,showLabels=true}={}) {
    nativeCharts.push(presentation.slides.items.length);
    return slide.charts.add('bar',{
        position:{left,top,width,height},categories,
        series:series.map(row=>({...row,values:row.values.map(value=>Number(value.toFixed(6)))})),
        hasLegend:series.length>1,legend:{position:'bottom',overlay:false,textStyle:{fontSize:23,fill:colors.ink}},
        barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:65},
        xAxis:{textStyle:{fontSize:22,fill:colors.ink},numberFormatCode:horizontal?format:undefined,line:{fill:colors.grid,width:1}},
        yAxis:{title:horizontal?undefined:{text:label,textStyle:{fontSize:22,fill:colors.ink}},min:0,max,
            numberFormatCode:format,textStyle:{fontSize:21,fill:colors.muted},majorGridlines:{fill:colors.grid,width:.7}},
        dataLabels:{showValue:showLabels,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:23,bold:true,fill:colors.ink}},
        chartFill:colors.paper,plotAreaFill:colors.paper,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
    });
}
function lines(slide, categories, series, left, width, ylabel, max) {
    nativeCharts.push(presentation.slides.items.length);
    return slide.charts.add('line',{
        position:{left,top:178,width,height:365},categories,
        series:series.map(row=>({...row,values:row.values.map(value=>Number(value.toFixed(6)))})),
        hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{fontSize:20,fill:colors.ink}},
        xAxis:{title:'Epoch',textStyle:{fontSize:12,fill:colors.muted}},
        yAxis:{title:ylabel,min:0,max,numberFormatCode:'0.00',textStyle:{fontSize:19,fill:colors.muted},majorGridlines:{fill:colors.grid,width:.6}},
        chartFill:colors.paper,plotAreaFill:colors.paper,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
    });
}

let slide;
slide=page('Automatic clothing\ncategory recognition', ['report/phase2/main.tex']);
text(slide,'Phase 2 / Project 11 / Section A',68,285,1120,45,27,false,colors.teal);
text(slide,'Small CNNs, controlled ablations and HOG fusion',68,367,1120,90,34);
text(slide,'Nakul    Sudeshhna    Puneet    Samarth    Neelanshu',68,575,1120,45,26,true);

slide=page('Learned features and similar garments', ['docs/phase2_literature_review.md','https://bottou.org/papers/lecun-98h','https://www.jmlr.org/papers/v15/srivastava14a.html','https://arxiv.org/abs/1711.05101']);
text(slide,'Can a small CNN improve classification\nand reduce upper-body confusion?',68,157,1120,125,40,true);
text(slide,'Phase 1 reference',68,338,530,45,30,true,colors.teal);
text(slide,'HOG + RBF SVM\n90.91% accuracy\n73.3% Shirt recall',68,402,530,160,30);
text(slide,'Phase 2 evidence',680,338,530,45,30,true,colors.teal);
text(slide,'Depth and dropout comparisons\nTraining-only augmentation\nHOG fusion with LR controls',680,402,530,165,28);

slide=page('Shared dataset and split membership', ['data/full_indices.npz','data/provenance.json','results/phase2/data_audit/data_audit.json']);
table(slide,[['Partition','Images','Purpose'],['Fit','48,000','Optimize CNN and LR weights'],['Validation','12,000','Select checkpoints and settings'],['Official test','10,000','Evaluate fixed comparison rows']],160,310,[280,210,654],28);
text(slide,'28 x 28 grayscale, 10 categories, original IDs preserved',68,515,1120,55,31,true,colors.teal);
text(slide,'The official public test was already examined in Phase 1.',68,605,1120,40,25,false,colors.muted);

slide=page('Selected CNN architecture', ['src/clothing_ml/phase2/models.py','results/phase2/full/runs/full_cnn3_s42_80a2edee/record.json']);
table(slide,data.architecture,150,435,[560,290,294],26);
text(slide,'Three blocks use 32, 64 and 128 channels. Embedding size is 64.',68,620,1120,40,25,true,colors.teal);

slide=page('Training and best-checkpoint selection', ['results/phase2/full/runs/full_cnn3_s42_80a2edee/history.csv','docs/phase2_training_protocol.md']);
lines(slide,data.history.map(row=>String(row.epoch)),[
    {name:'Fit loss',values:data.history.map(row=>row.fit_loss),line:{fill:colors.blue,width:2}},
    {name:'Validation loss',values:data.history.map(row=>row.val_loss),line:{fill:colors.orange,width:2}}
],60,560,'Cross-entropy',.85);
lines(slide,data.history.map(row=>String(row.epoch)),[
    {name:'Validation macro F1',values:data.history.map(row=>row.val_macro_f1),line:{fill:colors.teal,width:2}}
],664,550,'Macro F1',1);
text(slide,'AdamW 0.001, decay 0.0001, batch 128, patience 5',68,574,1120,45,27,true,colors.teal);
text(slide,'Best validation F1 at epoch 22. Training reaches the 25-epoch cap.',68,628,1120,35,24,false,colors.muted);

slide=page('Validation ablations', ['results/phase2/integration/cnn_candidates.csv','results/phase2/data_audit/augmentation_ablation.csv']);
table(slide,data.ablation_table,157,380,[460,210,240,234],26);
text(slide,'Same 48,000 fit / 12,000 validation IDs, seed 42',68,568,1120,45,29,true,colors.teal);
text(slide,'The later 50%-shift row stays outside the frozen selection set.',68,626,1120,36,24,false,colors.muted);

slide=page('Validation choice and frozen test rows', ['results/phase2/full/cnn_selection.json','results/phase2/full/freeze.json','results/phase2/full/test_access_log.md']);
table(slide,[['Decision','Rule'],['Checkpoint','Highest validation macro F1, earlier epoch on a tie'],['Main CNN','F1, then accuracy, then declared run order'],['LR control heads','C = 0.1 or 1, same fit / validation membership'],['Freeze','Pin CNN, three LR heads, artifacts and split hashes']],157,355,[315,829],26);
text(slide,'Phase 2 test access used the same fixed rows after a loader repair.',68,563,1120,80,28,true,colors.teal);

slide=page('Frozen official-test macro F1', ['results/phase2/integration/comparison.csv']);
bars(slide,data.comparison.map(row=>row.model),[{name:'Test macro F1',values:data.comparison.map(row=>row.test_macro_f1),fill:colors.teal}],{horizontal:true,format:'0.0000'});
text(slide,'Selected CNN: 92.14% accuracy and 0.9207 macro F1',68,621,1120,45,30,true,colors.teal);

slide=page('Paired CNN gain over HOG + SVM', ['results/phase2/full/evaluation/cnn_vs_svm_deltas.json','results/phase2/full/evaluation/paired_summary.json']);
table(slide,[['Measure','Observed gain','Paired bootstrap 95% interval'],['Accuracy','+1.23 percentage points','+0.73 to +1.75 percentage points'],['Macro F1','+0.0118','+0.0067 to +0.0170']],165,240,[300,330,514],27);
text(slide,'2,000 resamples of aligned test-ID pairs, seed 42',68,462,1120,50,30,true,colors.teal);
text(slide,'Intervals describe benchmark-image sampling variability.\nTraining-seed variation remains unmeasured.',68,548,1120,90,28);

slide=page('Category recall still matters', ['results/phase2/integration/per_class.csv']);
bars(slide,['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot'],[
    {name:'HOG + SVM',values:data.recall['HOG + SVM'],fill:colors.blue},
    {name:'CNN-3',values:data.recall['CNN-3'],fill:colors.teal}
],{label:'Recall',format:'0.0%',top:160,height:430,max:1.13,showLabels:false});
text(slide,'Shirt recall: 73.3% to 71.7%. 1,000 test images per true category.',68,629,1120,35,26,false,colors.muted);

slide=page('Shirt precision rises while recall falls', ['results/phase2/full/evaluation/per_class_metrics.csv','results/phase2/full/evaluation/label_pair_counts.csv']);
table(slide,data.shirt_table,160,240,[400,250,250,244],28);
text(slide,'True Shirts predicted as T-shirt: CNN 134, SVM 103',68,450,1120,55,31,true,colors.teal);
text(slide,'True T-shirts predicted as Shirt: CNN 64, SVM 104',68,534,1120,65,29);

slide=page('Paired upper-body prediction examples', ['results/phase2/full/evaluation/figure_example_ids.csv']);
for (let i=0;i<data.examples.length;i++) {
    const row=data.examples[i];
    const x=80+i*390;
    const group=['Only CNN correct','Only SVM correct','Both wrong'][i];
    text(slide,group+'\nTest ID '+row.sample_id,x,155,350,80,28,true,colors.teal);
    const bytes=await fs.readFile(path.join(outputDir,'assets','test_'+row.sample_id+'.png'));
    slide.images.add({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',
        alt:'Test image '+row.sample_id+', true '+row.true_category,fit:'contain',position:{left:x,top:270,width:230,height:230}});
    text(slide,'True: '+row.true_category+'\nCNN: '+row.cnn_category+'\nSVM: '+row.svm_category,x,518,350,100,25);
}
text(slide,'First sorted ID in each outcome group among true upper-body categories',68,645,1120,29,22,false,colors.muted);

slide=page('Hybrid feature fusion and controls', ['src/clothing_ml/phase2/fusion.py','results/phase2/full/fusion/fusion_candidates.csv']);
table(slide,[['Row','Features','Control purpose'],['Selected CNN','Own 10-way head','Practical learned-feature reference'],['Embedding + LR','64','Effect of the classical head'],['HOG + LR','1,296','Fixed-feature branch'],['Fusion + LR','1,360','Added HOG information']],155,355,[330,180,634],26);
text(slide,'Frozen encoder, ID-aligned features, fit-only scaler, same LR grid',68,561,1120,85,29,true,colors.teal);

slide=page('Fusion adds no aggregate improvement here', ['results/phase2/full/fusion/fusion_candidates.csv','results/phase2/integration/comparison.csv']);
const hybridRows=data.comparison.slice(1);
bars(slide,hybridRows.map(row=>row.model),[
    {name:'Validation',values:hybridRows.map(row=>row.val_macro_f1),fill:colors.blue},
    {name:'Official test',values:hybridRows.map(row=>row.test_macro_f1),fill:colors.teal}
],{top:165,height:420,format:'0.0000'});
text(slide,'Fusion validation F1 0.9261 vs embedding-only LR 0.9319',68,621,1120,48,29,true,colors.teal);

slide=page('Recorded CPU inference cost', ['results/phase2/full/evaluation/timing.json']);
table(slide,[['Pipeline','Median ms/image'],...data.comparison.map(row=>[row.model,row.ms_per_image.toFixed(3)])],157,360,[745,399],28);
text(slide,'Batch 128, 1,024 fixed IDs, 5 repeats, 1 CPU thread',68,559,1120,45,29,true,colors.teal);
text(slide,'Apple M3. Input conversion, features and prediction included.',68,624,1120,35,24,false,colors.muted);

slide=page('Model size uses model-specific units', ['results/phase2/full/evaluation/complexity.json']);
table(slide,data.complexity_table,155,380,[320,230,594],24);
text(slide,'CNN checkpoint: 673,525 bytes. Fusion also ships the frozen encoder.',68,594,1120,70,27,true,colors.teal);

slide=page('Integrated and reproducible evidence', ['results/phase2/integration/audit.json','docs/phase2_integration_nakul.md']);
text(slide,'31 prediction files checked',68,165,1120,72,46,true,colors.teal);
text(slide,'Original IDs and authoritative labels\nIndependent confusion-count metrics\nFrozen artifacts and selection hashes\nSelected checkpoint reload: 12,000 / 12,000 match',68,285,1120,250,32);
text(slide,'github.com/Nakul-Jaglan/aml-dl-t089',68,594,1120,50,30,true);

slide=page('Conclusions and measured limits', ['report/phase2/main.tex','results/phase2/integration/comparison.csv']);
text(slide,'92.14% test accuracy',68,160,1120,95,66,true,colors.teal);
text(slide,'Aggregate F1 improves and recorded CPU cost falls.\nShirt recall falls to 71.7%.\nFusion falls below the embedding-only control.',68,303,1120,190,32);
text(slide,'One training seed, one public split, small grayscale images, one CPU machine',68,573,1120,86,26,false,colors.muted);

slide=page('Selected references', ['docs/phase2_literature_review.md','report/phase2/references.bib']);
text(slide,'Xiao et al. (2017)  Fashion-MNIST\narxiv.org/abs/1708.07747\n\nLeCun et al. (1998)  Gradient-Based Learning Applied to Document Recognition\nbottou.org/papers/lecun-98h\n\nSrivastava et al. (2014)  Dropout\njmlr.org/papers/v15/srivastava14a.html',68,155,1140,390,26);
text(slide,'Loshchilov & Hutter (2019)  Decoupled Weight Decay Regularization\nDalal & Triggs (2005)  Histograms of Oriented Gradients',68,566,1140,88,24,false,colors.muted);

slide=page('Full results and offline demo', ['results/phase2/integration/comparison.csv','notebooks/01_phase2_master.ipynb']);
table(slide,data.results_table,153,360,[460,200,250,234],25);
text(slide,'Demo: original test ID 40, true Shirt. CNN correct, SVM predicts T-shirt.',68,565,1120,86,27,true,colors.teal);

if (presentation.slides.items.length!==notes.length) throw new Error('Expected 20 notes and slides');
for (let i=0;i<presentation.slides.items.length;i++) {
    const slide=presentation.slides.items[i];
    const preview=await presentation.export({slide,format:'png',scale:1});
    await fs.writeFile(path.join(buildDir,'slide-'+String(i+1).padStart(2,'0')+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
await (await PresentationFile.exportPptx(presentation)).save(path.join(buildDir,'candidate.pptx'));
await fs.writeFile(path.join(buildDir,'policy.json'),JSON.stringify({nativeTables,nativeCharts:[...new Set(nativeCharts)],slideCount:20},null,2));
await fs.writeFile(path.join(outputDir,'speaker_notes.md'),'# Phase 2 presenter notes\n\n'+notes.map((row,index)=>
    '## '+(index+1)+'. '+titles[index].replaceAll('\n',' ')+'\n\nPresenter: '+row.presenter+'\n\n'+row.notes+'\n\nSources\n'+sources[index].map(value=>'- '+value).join('\n')).join('\n\n')+'\n');
console.log('Rendered 20 slides and exported the candidate with editable charts, tables and Hinglish notes.');

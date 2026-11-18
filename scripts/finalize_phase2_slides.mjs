// Bundle the new charts' literal evidence into editable workbooks, then validate.
import path from 'node:path';
import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const root=process.cwd();
const skill=process.env.PRESENTATIONS_SKILL;
if (!skill) throw new Error('Set PRESENTATIONS_SKILL to the installed presentations skill folder.');
const python=process.env.PYTHON_EXECUTABLE || 'python3';
const build=path.join(root,'work/phase2_integration/deck');
const policy=JSON.parse(await fs.readFile(path.join(build,'policy.json'),'utf8'));
const portable=path.join(build,'portable.pptx');
execFileSync(python,[path.join(skill,'container_tools/materialize_literal_chart_workbooks.py'),
    path.join(build,'candidate.pptx'),portable,'--workspace',build,
    '--receipt',path.join(build,'chart_data_snapshot.json')],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:root,candidatePath:portable,
    finalPath:process.env.PHASE2_PPTX_OUTPUT || path.join(root,'presentation/phase2/Phase_2_Clothing_Category_Recognition.pptx'),
    pythonExecutable:python,
    integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
        ...policy.nativeTables.flatMap(n=>['--require-native-table-slide',String(n)])],
    explicitTotalSlideCount:20,requiredNativeChartOwnerSlides:policy.nativeCharts,
    requiredEmbeddedWorkbookChartOwnerSlides:policy.nativeCharts,
    requiredNativeTableOwnerSlides:policy.nativeTables,nativeChartTargetApplication:'powerpoint',
    materializeLiteralChartWorkbooks:false,fontPolicy:{basis:'design',families:['Arial']},
    verifyArtifactToolImport:true,receiptPath:path.join(root,'work/phase2_integration/deck_validation.json')});
console.log(JSON.stringify({finalPath:result.finalPath,package:result.packageIntegrity.status,
    layoutFindings:result.presentationLayout.findingCount,charts:result.nativeChartValidation.passed,
    importPassed:result.firstPartyImport.passed}));

"""Reproduce published source event codes, using narrative text alone.

ITA metrics measure agreement with an autocoder, not independently verified
hazard truth. This fixed benchmark requires no new human labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABELS=list('1234567')
SEED=2026
CONFIG={'seed':SEED,'scope':'NAICS starts with 4881; unverified ground activity',
        'predictor':'narrative only','target':'published OIICS 2.x event division, first digit',
        'training':'SIR 2015-2020','validation':'SIR 2021','temporal_holdout':'SIR 2022',
        'excluded':'Previously inspected groups; unspecified labels; SIR groups reaching 2023+; invalid dates/blank text',
        'selection_metric':'macro F1 over divisions actually represented in 2021 validation gold source codes',
        'features':{'ngrams':[1,2],'min_df':2,'max_features':30000,'sublinear_tf':True},
        'regularization_C':1.0,'class_weight':'balanced','bootstrap_resamples':500,
        'external_comparisons':'ITA actual 2023/2024 dates, coded with BLS SOII autocoder OIICS 2.01; not independent gold'}

def model_specs():
    def tfidf():
        return TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,strip_accents='unicode',sublinear_tf=True)
    return {
        'majority':DummyClassifier(strategy='most_frequent'),
        'tfidf_logistic':Pipeline([('tfidf',tfidf()),('classifier',LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,random_state=SEED))]),
        'tfidf_linear_svm':Pipeline([('tfidf',tfidf()),('classifier',LinearSVC(C=1.0,class_weight='balanced',random_state=SEED,max_iter=10000))]),
    }

def inputs(name, frame):
    # Audit/source fields never reach the vectorizer or classifier.
    return np.zeros((len(frame),1)) if name=='majority' else frame.narrative.astype(str)

def metrics(frame, pred):
    y=frame.event_division.to_numpy();present=sorted(set(y))
    return {'records':len(frame),'event_groups':frame.event_group.nunique(),
            'source_reference_accuracy':float(accuracy_score(y,pred)),
            'macro_f1_supported_divisions':float(f1_score(y,pred,labels=present,average='macro',zero_division=0)),
            'macro_f1_all_seven_divisions':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,average='weighted',zero_division=0)),
            'supported_reference_divisions':present,
            'unsupported_reference_divisions':sorted(set(LABELS)-set(present)),
            'reference_division_counts':frame.event_division.value_counts().sort_index().to_dict(),
            'predicted_division_counts':pd.Series(pred).value_counts().sort_index().to_dict(),
            'reference_interpretation':'Agreement with BLS autocoder predictions' if frame.source.str.startswith('OSHA_ITA').all() else 'Reproduction of OSHA published event divisions; individual label provenance not verified'}

def bootstrap(frame,pred):
    groups=frame.event_group.to_numpy();unique=np.unique(groups)
    if len(unique)<10:return {'status':'Not estimated: fewer than 10 conservative groups'}
    group_index={g:i for i,g in enumerate(unique)};label_index={l:i for i,l in enumerate(LABELS)}
    tables=np.zeros((len(unique),7,7),dtype=np.int64)
    for g,y,p in zip(groups,frame.event_division,pred):tables[group_index[g],label_index[y],label_index[p]]+=1
    supported=[label_index[l] for l in sorted(frame.event_division.unique())]
    rng=np.random.default_rng(SEED);scores=[]
    for _ in range(CONFIG['bootstrap_resamples']):
        matrix=tables[rng.integers(0,len(unique),size=len(unique))].sum(axis=0)
        tp=matrix.diagonal();den=matrix.sum(axis=0)+matrix.sum(axis=1)
        f1=np.divide(2*tp,den,out=np.zeros(7,dtype=float),where=den!=0)
        scores.append([float(tp.sum()/matrix.sum()),float(f1[supported].mean())])
    bounds=np.quantile(np.asarray(scores),[.025,.975],axis=0)
    return {'method':'Percentile bootstrap resampling conservative event groups; 500 draws; seed 2026',
            'accuracy_95_percent_interval':bounds[:,0].tolist(),
            'supported_macro_f1_95_percent_interval':bounds[:,1].tolist(),
            'caution':'Conditional sampling variability only; does not quantify scope error, missing reports, coding error or systematic autocoder bias'}

def verify(frame):
    required={'record_id','narrative','event_division','source','event_group','split','text_sha256','event_date','naics','source_event_code'}
    if not required.issubset(frame):raise ValueError('Missing fields: '+str(required-set(frame)))
    if frame.record_id.duplicated().any():raise ValueError('Duplicate record IDs')
    used=frame[frame.split.ne('excluded')]
    if not used.event_division.isin(LABELS).all() or used.narrative.eq('').any():raise ValueError('Invalid target or blank text')
    if not used.naics.str.startswith('4881').all():raise ValueError('Scope differs from declared industry proxy')
    if not used.event_division.eq(used.source_event_code.str[0]).all():raise ValueError('Target was not derived from its actual source code')
    splits=['train','validation','SIR_2022_temporal_test']
    sets=[set(used.loc[used.split.eq(s),'event_group']) for s in splits]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise AssertionError('Event-group leakage among SIR partitions')
    texts=[set(used.loc[used.split.eq(s),'text_sha256']) for s in splits]
    if any(texts[i]&texts[j] for i in range(3) for j in range(i)):raise AssertionError('Exact narrative leakage among SIR partitions')
    sir_groups=set.union(*sets)
    external=used[used.source.str.startswith('OSHA_ITA')]
    if sir_groups&set(external.event_group):raise AssertionError('SIR group in external comparison')
    years=pd.to_datetime(used.event_date).dt.year
    if not years[used.split.eq('train')].le(2020).all():raise AssertionError('Training contains future reports')
    if not years[used.split.eq('validation')].eq(2021).all():raise AssertionError('Validation date mismatch')
    if not years[used.split.eq('SIR_2022_temporal_test')].eq(2022).all():raise AssertionError('Temporal test date mismatch')
    if len(used[used.split.eq('train')])<50 or used.loc[used.split.eq('train'),'event_division'].nunique()<2:raise ValueError('Insufficient training records/classes')
    return {'status':'passed','distinct_record_ids':True,'source_target_integrity':True,
            'predictor_is_narrative_only':True,'sir_groups_and_exact_text_disjoint':True,'sir_external_groups_disjoint':True,
            'actual_date_boundaries_verified':True,'scope_is_industry_proxy_not_ground_eligibility':True,
            'external_2023_2024_shared_groups':len(set(external.loc[external.source.eq('OSHA_ITA_2023'),'event_group'])&set(external.loc[external.source.eq('OSHA_ITA_2024'),'event_group']))}

def run(data_path,output_dir):
    data_path=Path(data_path);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(data_path,dtype=str,keep_default_na=False)
    checks=verify(frame)
    train=frame[frame.split.eq('train')].copy();val=frame[frame.split.eq('validation')].copy()
    data_hash=hashlib.sha256(data_path.read_bytes()).hexdigest()
    config_hash=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
    validation={};trained={};captured_warnings=[]
    for name,model in model_specs().items():
        start=time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(inputs(name,train),train.event_division)
        captured_warnings.extend({'model':name,'category':type(w.message).__name__,'message':str(w.message)} for w in caught)
        pred=model.predict(inputs(name,val));validation[name]=metrics(val,pred)
        validation[name]['fit_seconds']=time.time()-start
        validation[name]['tfidf_features']=len(model.named_steps['tfidf'].vocabulary_) if name!='majority' else 0
        trained[name]=model
    # Fixed model specifications. Only the declared 2021 validation metric selects the model.
    selected=max(['tfidf_logistic','tfidf_linear_svm'],key=lambda n:validation[n]['macro_f1_supported_divisions'])
    decision={'selected_model':selected,'selection_metric':CONFIG['selection_metric'],
              'configuration_sha256':config_hash,'data_sha256':data_hash,'validation':validation,
              'test_metrics_used_for_selection':False,'refit_on_validation':False}
    (output_dir/'validation_selection.json').write_text(json.dumps(decision,indent=2))
    evaluations={};predictions=[];class_rows=[]
    for split in ['SIR_2022_temporal_test','ITA_2023_autocoder_comparison','ITA_2024_autocoder_comparison']:
        test=frame[frame.split.eq(split)].copy()
        if test.empty:
            evaluations[split]={'status':'not evaluated: no eligible records'};continue
        evaluations[split]={}
        for name in [selected,'majority']:
            pred=trained[name].predict(inputs(name,test))
            result=metrics(test,pred);result['group_bootstrap']=bootstrap(test,pred)
            evaluations[split][name]=result
            report=classification_report(test.event_division,pred,labels=LABELS,output_dict=True,zero_division=0)
            for label in LABELS:class_rows.append({'split':split,'model':name,'event_division':label,**report[label]})
            pd.DataFrame(confusion_matrix(test.event_division,pred,labels=LABELS),index=LABELS,columns=LABELS).to_csv(output_dir/(split+'_'+name+'_confusion.csv'),index_label='reference_division')
            p=test[['record_id','source','event_date','event_group','text_sha256','event_division']].copy()
            p['model']=name;p['predicted_division']=pred;p['evaluation_split']=split
            p['reference_type']='autocoder' if split.startswith('ITA') else 'published_OSHA_code'
            predictions.append(p)
    pd.concat(predictions,ignore_index=True).to_csv(output_dir/'evaluated_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(class_rows).to_csv(output_dir/'per_class_results.csv',index=False,encoding='utf-8-sig')
    result={'study_status':'Completed fixed public-source-code classification benchmark',
            'config':CONFIG,'training_records':len(train),'training_event_groups':train.event_group.nunique(),
            'training_division_counts':train.event_division.value_counts().sort_index().to_dict(),
            'selection':decision,'evaluations':evaluations,'verification':checks,'warnings':captured_warnings,
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__},
            'limits':['Industry proxy does not establish ground-worker activity.','Small SIR validation and temporal test samples; rare divisions poorly evaluated.',
                      'ITA labels are machine-generated; agreement does not establish hazard accuracy.','No pilot/worker interviews or independent human labels.',
                      'NTSB/ASRS have no harmonized OIICS occupational-hazard targets; no performance metrics calculated on them.',
                      'Test has now been evaluated; subsequent tuning requires a fresh evaluation protocol.']}
    (output_dir/'benchmark_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output_dir/'workflow_verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps({'selected':selected,'training_records':len(train),'validation_records':len(val),
                      'evaluations':{s:{m:{k:v for k,v in r.items() if k in ['records','source_reference_accuracy','macro_f1_supported_divisions','supported_reference_divisions']} for m,r in rs.items()} for s,rs in evaluations.items()},
                      'warnings':captured_warnings},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data/private/sourcecoded_records.csv');parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'local_outputs')
    args=parser.parse_args();run(args.data,args.output_dir)

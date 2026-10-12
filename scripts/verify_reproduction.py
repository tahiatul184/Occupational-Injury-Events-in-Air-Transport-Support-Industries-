"""Compare a rerun with the frozen aggregate and prediction checksums."""
import argparse,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'local_outputs')
    args=p.parse_args();out=args.output_dir
    expected=json.loads((ROOT/'results/benchmark_results.json').read_text())
    actual=json.loads((out/'benchmark_results.json').read_text())
    for key in ['config','training_records','training_event_groups','training_division_counts','evaluations','verification','warnings']:
        assert actual[key]==expected[key],key+' differs from the frozen benchmark'
    for key in ['selected_model','selection_metric','configuration_sha256','data_sha256','test_metrics_used_for_selection','refit_on_validation']:
        assert actual['selection'][key]==expected['selection'][key],key+' differs'
    for model,reference in expected['selection']['validation'].items():
        for key,value in reference.items():
            if key!='fit_seconds':assert actual['selection']['validation'][model][key]==value,(model,key)
    for f in (ROOT/'results').glob('*_confusion.csv'):
        assert (out/f.name).read_bytes()==f.read_bytes(),f.name+' differs'
    assert (out/'per_class_results.csv').read_bytes()==(ROOT/'results/per_class_results.csv').read_bytes()
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    prediction=out/'evaluated_predictions.csv'
    sha=hashlib.sha256(prediction.read_bytes()).hexdigest()
    assert sha==manifest['evaluated_predictions_sha256'],'Prediction checksum differs'
    with prediction.open(encoding='utf-8-sig',newline='') as f:rows=sum(1 for _ in csv.DictReader(f))
    assert rows==16858,rows
    report={'status':'passed','evaluated_prediction_rows':rows,'evaluated_predictions_sha256':sha,
            'aggregate_metrics_match':True,'bootstrap_intervals_match':True,'six_confusion_matrices_match':True,
            'configuration_sha256':actual['selection']['configuration_sha256'],
            'cohort_sha256':actual['selection']['data_sha256'],'new_validation_cohort':False,'new_human_labels':0,
            'runtime':actual['runtime']}
    (out/'reproduction_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

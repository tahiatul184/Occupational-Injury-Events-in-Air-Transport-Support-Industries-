"""Reconstruct the frozen cohort from its SIR checkpoint and original ITA archives."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.source_preparation import file_sha,prepare_ita,employer_day_key,combine_cohort

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sir-checkpoint',type=Path,required=True)
    p.add_argument('--ita-2023',type=Path,required=True)
    p.add_argument('--ita-2024',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/private')
    args=p.parse_args()
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    paths=[args.sir_checkpoint,args.ita_2023,args.ita_2024]
    hashes=[manifest['sir_checkpoint_sha256']]+[x['sha256'] for x in manifest['ita_archives']]
    for path,expected in zip(paths,hashes):
        if file_sha(path)!=expected:
            raise SystemExit('Input checksum differs from the frozen study snapshot: '+path.name)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sir=pd.read_csv(args.sir_checkpoint,dtype=str,keep_default_na=False)
    frames=[];audits=[]
    for path in [args.ita_2023,args.ita_2024]:
        frame,audit=prepare_ita(path,args.output_dir,employer_day_key)
        frames.append(frame);audits.append(audit)
    cohort=combine_cohort(sir,frames,args.output_dir)
    actual=file_sha(args.output_dir/'sourcecoded_records.csv')
    if actual!=manifest['cohort_sha256']:
        raise SystemExit('Reconstructed cohort differs from the frozen cohort; do not claim exact replication.')
    audit={'status':'passed','cohort_sha256':actual,'candidate_reports':len(cohort),
           'retained_reports':int(cohort.split.ne('excluded').sum()),
           'split_counts':cohort.split.value_counts().to_dict(),'ita_audits':audits,
           'sir_input':'Historical 414-row checkpoint; no new full SIR extraction','new_human_labels':0}
    (args.output_dir/'cleaning_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='ita_audits'},indent=2))
if __name__=='__main__':main()

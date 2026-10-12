"""Original preparation functions retained for the fixed SIR/ITA benchmark."""
from __future__ import annotations
import codecs, io, hashlib, json, re, zipfile
from pathlib import Path
import pandas as pd

LABELS = {
    '1': 'Violence and other injuries by persons or animals',
    '2': 'Transportation incidents',
    '3': 'Fires and explosions',
    '4': 'Falls, slips, trips',
    '5': 'Exposure to harmful substances or environments',
    '6': 'Contact with objects and equipment',
    '7': 'Overexertion and bodily reaction',
}


WORKER_RE = re.compile(r'\b(?:ground crew|ground personnel|ground worker|ramp (?:agent|worker|personnel)|baggage handler|airport (?:employee|worker)|marshaller|wing ?walker|line ?man|line service|fuel(?:l)?er|refueller|refueler|ground handling)\b', re.I)


AIRBORNE_RE = re.compile(r'\b(?:flight attendant|in[- ]flight|during flight|cruise|turbulence)\b', re.I)


STATE_NAMES=['ALABAMA','ALASKA','ARIZONA','ARKANSAS','CALIFORNIA','COLORADO','CONNECTICUT','DELAWARE','FLORIDA','GEORGIA','HAWAII','IDAHO','ILLINOIS','INDIANA','IOWA','KANSAS','KENTUCKY','LOUISIANA','MAINE','MARYLAND','MASSACHUSETTS','MICHIGAN','MINNESOTA','MISSISSIPPI','MISSOURI','MONTANA','NEBRASKA','NEVADA','NEWHAMPSHIRE','NEWJERSEY','NEWMEXICO','NEWYORK','NORTHCAROLINA','NORTHDAKOTA','OHIO','OKLAHOMA','OREGON','PENNSYLVANIA','RHODEISLAND','SOUTHCAROLINA','SOUTHDAKOTA','TENNESSEE','TEXAS','UTAH','VERMONT','VIRGINIA','WASHINGTON','WESTVIRGINIA','WISCONSIN','WYOMING','DISTRICTOFCOLUMBIA','PUERTORICO']


STATE_ABBREVIATIONS=['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC','PR']


def employer_day_key(name,date,city='',state=''):
    name=re.sub(r'[^a-z0-9]+','',str(name).lower())
    city=re.sub(r'[^a-z0-9]+','',str(city).lower())
    state=re.sub(r'[^A-Z0-9]+','',str(state).upper())
    state=dict(zip(STATE_NAMES,STATE_ABBREVIATIONS)).get(state,state)
    return 'EMPLOYER_DAY_'+sha('|'.join([name,city,state,date]))[:24] if name and date else ''


def norm(x):
    return re.sub(r'\s+', ' ', str(x)).strip()


def sha(x):
    return hashlib.sha256(str(x).encode('utf-8')).hexdigest()


def file_sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def join_narratives(row, cols):
    values, seen = [], set()
    for c in cols:
        value = norm(row[c])
        if value and value.lower() not in seen:
            values.append(value); seen.add(value.lower())
    return ' '.join(values)


def clean_code(x):
    return re.sub(r'\.0$', '', str(x).strip())


def linked_groups(frame):
    parent = {rid: rid for rid in frame.record_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    for col in ['original_event_group', 'text_sha256', 'employer_day_key']:
        for key, block in frame.groupby(col, sort=False):
            if not key: continue
            ids = block.record_id.tolist()
            for rid in ids[1:]: union(ids[0], rid)
    return ['SOURCECODE_GROUP_' + sha(find(rid))[:24] for rid in frame.record_id]


def prepare_ita(archive, out, employer_day_key):
    year = re.search(r'Data_(202[34])', archive.name).group(1)
    source = 'OSHA_ITA_' + year
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(members) != 1: raise ValueError('Expected one ITA CSV')
        member = members[0]
        decode_counts={'fallback_bytes':0,'undefined_cp1252_bytes_preserved':0}
        def preserve_invalid_utf8(exc):
            chars=[]
            for byte in exc.object[exc.start:exc.end]:
                decode_counts['fallback_bytes']+=1
                try:chars.append(bytes([byte]).decode('cp1252','strict'))
                except UnicodeDecodeError:
                    # Retain the byte's Latin-1 code point rather than deleting it.
                    chars.append(chr(byte));decode_counts['undefined_cp1252_bytes_preserved']+=1
            return ''.join(chars),exc.end
        handler_name='ita_preserve_bytes_'+year
        codecs.register_error(handler_name,preserve_invalid_utf8)
        encoding='UTF-8 valid sequences; undecodable bytes mapped with Windows-1252; undefined Windows-1252 bytes retained as Latin-1 code points'
        with z.open(member) as f:
            decoded=io.TextIOWrapper(f,encoding='utf-8-sig',errors=handler_name,newline='')
            chunks = pd.read_csv(decoded, dtype=str, keep_default_na=False, chunksize=50000)
            rows, total, blank_events, aviation, support = [], 0, 0, 0, 0
            event_dates, invalid_dates = [], 0
            for chunk in chunks:
                chunk.columns = chunk.columns.str.lower()
                required = {'event_code_pred','event_title_pred','naics_code','date_of_incident','id','job_description'}
                if not required.issubset(chunk): raise ValueError('ITA schema changed')
                narrative_cols = ['new_incident_description','new_nar_before_incident','new_nar_what_happened','new_nar_injury_illness','new_nar_object_substance']
                if not set(narrative_cols).issubset(chunk): raise ValueError('Missing narrative fields')
                total += len(chunk)
                blank_events += int(chunk.event_code_pred.str.strip().eq('').sum())
                aviation += int(chunk.naics_code.str.match(r'^(481|4881)').sum())
                mask = chunk.naics_code.str.startswith('4881')
                support += int(mask.sum())
                dates = pd.to_datetime(chunk.date_of_incident, errors='coerce', format='mixed')
                invalid_dates += int(dates.isna().sum())
                if dates.notna().any(): event_dates += [dates.min(), dates.max()]
                for idx, row in chunk.loc[mask].iterrows():
                    narrative = join_narratives(row, narrative_cols)
                    date = dates.loc[idx].strftime('%Y-%m-%d') if pd.notna(dates.loc[idx]) else ''
                    rid = source + '_' + str(idx + 2) + '_' + sha(narrative)[:10]
                    code = clean_code(row.event_code_pred)
                    rows.append({
                        'record_id':rid, 'source':source, 'source_record_id':row.id,
                        'source_row_number':idx + 2, 'event_date':date, 'naics':row.naics_code,
                        'job_description':norm(row.job_description), 'narrative':narrative,
                        'text_sha256':sha(narrative.lower()), 'source_event_code':code,
                        'source_event_title':row.event_title_pred, 'event_division':code[:1] if code[:1] in LABELS else '',
                        'label_provenance':'BLS SOII autocoder; OIICS 2.01; two-digit nonterminal predictions',
                        'coding_version':'2.01 (source documentation)', 'original_event_group':source + '_ROW_' + str(idx),
                        'employer_day_key':employer_day_key(row.establishment_name,date,row.city,row.state),
                        'development_only':False, 'latest_sir_group_year':'',
                        'worker_role_text_hit':bool(WORKER_RE.search(narrative + ' ' + row.job_description)),
                        'airborne_text_hit':bool(AIRBORNE_RE.search(narrative + ' ' + row.job_description)),
                        'scope_basis':'NAICS starts with 4881; worker ground-operation eligibility unverified',
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out / (source.lower() + '_support_candidates.csv'),index=False,encoding='utf-8-sig')
    audit = {'archive':archive.name,'archive_sha256':file_sha(archive),'archive_bytes':archive.stat().st_size,
             'csv_member':member,'encoding':encoding,'decode_counts':decode_counts,'total_records':total,
             'aviation_industry_records':aviation,'support_industry_records':support,
             'blank_event_codes_all_records':blank_events,'blank_event_codes_support_records':int(df.source_event_code.eq('').sum()),
             'actual_date_min':min(event_dates).strftime('%Y-%m-%d'),'actual_date_max':max(event_dates).strftime('%Y-%m-%d'),
             'invalid_actual_dates':invalid_dates,'support_actual_year_counts':df.event_date.str[:4].value_counts().to_dict(),
             'event_division_counts_support':df.event_division.value_counts().to_dict(),
             'blank_support_narratives':int(df.narrative.eq('').sum()),'unique_source_ids_support':int(df.source_record_id.nunique()),
             'label_type':'machine-generated reference codes; not independently adjudicated gold'}
    print('ITA_AUDIT',json.dumps(audit),flush=True)
    return df, audit


def combine_cohort(sir,ita_frames,out):
    fields=['record_id','source','source_record_id','source_row_number','event_date','naics','job_description','narrative','text_sha256','source_event_code','source_event_title','event_division','label_provenance','coding_version','original_event_group','employer_day_key','development_only','latest_sir_group_year','worker_role_text_hit','airborne_text_hit','scope_basis']
    frames=[sir[fields]]+[f[fields] for f in ita_frames]
    combined=pd.concat(frames,ignore_index=True)
    if combined.record_id.duplicated().any():raise AssertionError('Duplicate record IDs')
    combined['event_group']=linked_groups(combined)
    combined['year']=pd.to_datetime(combined.event_date,errors='coerce',format='mixed').dt.year
    combined['split']='excluded'
    combined['exclusion_reason']=''
    source_sir=combined.source.eq('OSHA_SIR')
    latest_sir=pd.to_numeric(combined.latest_sir_group_year,errors='coerce').groupby(combined.event_group).transform('max')
    dev_groups=set(combined.loc[combined.development_only.astype(str).str.lower().isin(['true','1','yes']),'event_group'])
    # All connected SIR reports take the latest SIR group's year to prevent temporal overlap.
    combined['sir_group_year']=latest_sir
    valid=combined.event_division.isin(LABELS)&combined.narrative.ne('')&combined.year.notna()
    combined.loc[source_sir&valid&combined.sir_group_year.le(2020),'split']='train'
    combined.loc[source_sir&valid&combined.sir_group_year.eq(2021),'split']='validation'
    combined.loc[source_sir&valid&combined.sir_group_year.eq(2022),'split']='SIR_2022_temporal_test'
    combined.loc[~source_sir&valid&combined.year.eq(2023)&combined.source.eq('OSHA_ITA_2023'),'split']='ITA_2023_autocoder_comparison'
    combined.loc[~source_sir&valid&combined.year.eq(2024)&combined.source.eq('OSHA_ITA_2024'),'split']='ITA_2024_autocoder_comparison'
    combined.loc[~valid,'exclusion_reason']='Missing/unspecified event division, blank narrative or invalid date'
    version_shift=source_sir & latest_sir.ge(2023)
    combined.loc[version_shift,['split','exclusion_reason']]=['excluded','SIR group reaches 2023+ coding transition']
    combined.loc[combined.event_group.isin(dev_groups),['split','exclusion_reason']]=['excluded','Previously inspected development group']
    used_sir_groups=set(combined.loc[source_sir&combined.split.ne('excluded'),'event_group'])
    shared=~source_sir&combined.event_group.isin(used_sir_groups)
    combined.loc[shared,['split','exclusion_reason']]=['excluded','Shares a conservative group with SIR benchmark']
    earlier_linked=source_sir & combined.split.isin(['validation','SIR_2022_temporal_test']) & combined.year.ne(combined.sir_group_year)
    combined.loc[earlier_linked,['split','exclusion_reason']]=['excluded','Earlier report linked to a newer SIR holdout group']
    unexplained=combined.split.eq('excluded')&combined.exclusion_reason.eq('')
    combined.loc[unexplained,'exclusion_reason']='Actual date outside declared source comparison year'
    combined.to_csv(out/'sourcecoded_records.csv',index=False,encoding='utf-8-sig')
    return combined

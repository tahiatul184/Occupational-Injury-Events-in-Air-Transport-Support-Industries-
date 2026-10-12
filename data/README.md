# Input data and source snapshots

This directory contains metadata only. Report-level CSVs and raw archives belong in `data/private/`, which Git ignores.

For the direct rerun, supply `sourcecoded_records.csv` from the existing prepared study package. The exact SHA-256 is `bea81ee7c47e545f1a80016c07e36fc69fa8c052a0ec87e78897a6bb3ffc79fd`. It contains 9,264 candidates, 8,665 retained reports and 599 exclusions. The complete 26-column schema is in `source_manifest.json`.

Essential fields are `record_id`, `narrative`, `event_division`, `source`, `event_group`, `split`, `text_sha256`, `event_date`, `naics` and `source_event_code`. The predictor is narrative text alone. Dates, source codes, identifiers, occupations and grouping fields are audit metadata.

Full cohort reconstruction additionally requires the checksum-matching historical SIR checkpoint and both original ITA ZIPs. The checkpoint preserves source rows, earlier groups and previous development exclusions. It cannot be replaced by an arbitrary current SIR download without redefining the experiment.

Official source portals: [OSHA Severe Injury Reports](https://www.osha.gov/severeinjury/) and [OSHA ITA data](https://www.osha.gov/itadata). Exact ITA filenames, URLs, sizes and hashes are recorded in the manifest. Original source access/audit date: 11 October 2026; the original SIR download date was not recovered.

Source reuse and redistribution conditions have not been confirmed for a new report-level deposit. This repository makes no claim of an independently labeled ground-worker dataset. No worker or pilot contact is needed to reuse the existing benchmark inputs.

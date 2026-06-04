# Gemma4Tunnel cloud implementation steps

Target repositories:

- GitHub source repo: `kilickursat/gemma4tunnel`
- Hugging Face model repo: `TUSTResearcher/gemma4tunnel`
- Hugging Face dataset repo: `TUSTResearcher/gemma4tunnel-data`
- Recommended Hugging Face Space repo: `TUSTResearcher/gemma4tunnel-demo`

Upload the extracted contents of this folder to the root of the GitHub repo. Do not upload this zip as one file.

Required GitHub Actions secrets:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `HF_TOKEN`
- `HF_USERNAME` = `TUSTResearcher`
- `ZOTERO_API_KEY`
- `ZOTERO_LIBRARY_TYPE` = `user`
- `ZOTERO_LIBRARY_ID`
- `ZOTERO_COLLECTION_KEY` optional
- `UNPAYWALL_EMAIL`

Run order:

1. Actions -> Run Gemma4Tunnel on Modal -> action=`ingest`, dataset_repo=`TUSTResearcher/gemma4tunnel-data`, model_repo=`TUSTResearcher/gemma4tunnel`.
2. Check `TUSTResearcher/gemma4tunnel-data` and review `license_audit.csv`.
3. Actions -> Run Gemma4Tunnel on Modal -> action=`train`, max_steps=`20`.
4. If smoke test succeeds, run training again with larger max_steps.
5. Actions -> Sync HF Space Demo -> space_repo=`TUSTResearcher/gemma4tunnel-demo` or your actual Space repo id.

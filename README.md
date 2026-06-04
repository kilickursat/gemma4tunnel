# Gemma4Tunnel Cloud-Only Starter

This repository is designed for a no-local-command workflow:

1. Upload these files to a GitHub repository using the GitHub web UI.
2. Add GitHub Actions secrets for Modal, Hugging Face, and Zotero.
3. Click **Run workflow** in GitHub Actions.
4. GitHub Actions calls Modal.
5. Modal pulls allowed Zotero PDFs through the Zotero Web API, builds a licensed dataset, fine-tunes a Gemma 4 LoRA adapter with Unsloth, and uploads the outputs to Hugging Face.

The pipeline does **not** upload your Zotero PDFs to Hugging Face. It uploads only curated JSONL training examples, source metadata, and the LoRA adapter.

## Repository layout

```text
.
├── .github/workflows/run-modal.yml       # Manual GitHub Actions button
├── modal_gemma4tunnel_zotero.py          # Modal ingestion + triage + training
├── cards/
│   ├── DATASET_CARD_TEMPLATE.md
│   └── MODEL_CARD_TEMPLATE.md
└── space/
    ├── README.md
    ├── app.py
    └── requirements.txt
```

## Cloud accounts needed

- Zotero account with synced metadata and, ideally, Zotero-stored PDFs.
- Hugging Face account with a write token.
- Modal account with token id and token secret.
- GitHub repository to host this code.

## GitHub Actions secrets

In your GitHub repository, go to:

**Settings → Secrets and variables → Actions → New repository secret**

Add:

```text
MODAL_TOKEN_ID             # Modal token id
MODAL_TOKEN_SECRET         # Modal token secret
HF_TOKEN                   # Hugging Face write token
HF_USERNAME                # Your Hugging Face username or org name
ZOTERO_API_KEY             # Zotero API key
ZOTERO_LIBRARY_TYPE        # user or group
ZOTERO_LIBRARY_ID          # numeric Zotero userID or groupID
ZOTERO_COLLECTION_KEY      # optional; leave empty to scan all top-level items
UNPAYWALL_EMAIL            # email for Unpaywall API queries
```

### Zotero details

- For a personal library, use `ZOTERO_LIBRARY_TYPE=user` and your numeric Zotero user ID.
- For a group library, use `ZOTERO_LIBRARY_TYPE=group` and the group ID.
- The collection key is optional. If you want safer control, create a Zotero collection called something like `Gemma4Tunnel-candidates` and put only tunnelling/TBM papers there.

## Run from the cloud

Open your GitHub repository, then:

**Actions → Run Gemma4Tunnel on Modal → Run workflow**

Recommended first run:

```text
action: ingest
base_model: unsloth/gemma-4-12b-it
dataset_repo: YOUR_HF_USERNAME/gemma4tunnel-data
model_repo: YOUR_HF_USERNAME/gemma4tunnel
max_steps: 20
```

Then check your Hugging Face dataset repo. You should see:

```text
train.jsonl
validation.jsonl
license_audit.csv
README.md
```

Then run:

```text
action: train
max_steps: 20
```

After the smoke test succeeds, run a longer job:

```text
action: train
max_steps: 1000
max_seq_length: 4096
gpu: H100
```

You can also run:

```text
action: all
```

This performs ingestion then training in one run.

## License triage policy used by this starter

The triage is conservative. It uses Zotero metadata, DOI, Crossref metadata, and Unpaywall metadata.

### Allowed for public open-model training by default

- CC0 / public domain
- CC BY
- CC BY-SA, with ShareAlike warning in the audit
- Your own papers/notes/manual text if you explicitly tag them in Zotero with `gemma4tunnel-own-work`

### Routed to private RAG by default

- CC BY-NC
- CC BY-NC-SA
- Bronze OA / free-to-read with no explicit reuse license
- Green OA copies with no explicit reuse license
- Publisher PDFs with all-rights-reserved terms
- Anything unknown or missing DOI/license evidence

### Excluded from public training by default

- CC BY-ND
- CC BY-NC-ND
- Closed/all-rights-reserved PDFs
- Missing/unclear license

The output `license_audit.csv` explains the decision per Zotero item.

## Important note

A non-commercial research purpose does not automatically make every PDF safe for a public model release. This starter therefore treats ambiguous material as RAG-only. You can override decisions in Zotero by adding controlled tags, but you should only do that when you have written permission or clear license evidence.

## Optional: Hugging Face Space demo

The `space/` folder is a simple Gradio demo skeleton. Create a Hugging Face Space manually in the web UI and upload the three files in `space/`.

Edit `space/app.py` first:

```python
BASE_MODEL = "google/gemma-4-12B-it"
ADAPTER_MODEL = "YOUR_HF_USERNAME/gemma4tunnel"
```

A 12B model usually needs GPU hardware for acceptable speed.

## What this starter intentionally does not do

- It does not publish source PDFs.
- It does not train on PDFs with unclear reuse rights.
- It does not guarantee legal compliance; it gives a conservative technical workflow and audit trail.
- It does not replace a private RAG system for exact citation-heavy answers.

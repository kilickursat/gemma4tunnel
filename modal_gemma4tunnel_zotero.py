"""Cloud-only Gemma4Tunnel pipeline.

Run from GitHub Actions with:
    modal run modal_gemma4tunnel_zotero.py --action ingest ...

The GitHub Actions runner only launches the job. Modal performs the actual PDF
fetching, dataset creation, GPU training, and Hugging Face upload.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

APP_NAME = "gemma4tunnel-cloud-zotero"
VOLUME_PATH = Path("/vol")
DATA_DIR = VOLUME_PATH / "data"
PDF_DIR = VOLUME_PATH / "pdfs"
OUT_DIR = VOLUME_PATH / "hf_upload"

# The GPU is read at Modal app construction time by the GitHub Actions runner.
# Set the MODAL_GPU env var from the workflow input. Default: H100.
MODAL_GPU = os.environ.get("MODAL_GPU", "H100")

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "requests>=2.32.0",
        "pymupdf>=1.24.0",
        "huggingface_hub>=0.31.0",
        "datasets>=3.0.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
    )
)

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.6.0",
        "accelerate>=1.7.0",
        "datasets>=3.6.0",
        "peft>=0.15.0",
        "trl>=0.18.0",
        "bitsandbytes>=0.45.0",
        "sentencepiece>=0.2.0",
        "protobuf>=5.0.0",
        "huggingface_hub>=0.31.0",
        "torchvision>=0.21.0",
        "librosa>=0.10.2",
        "unsloth",
        "unsloth_zoo",
    )
    # Gemma 4 12B uses the new gemma4_unified architecture.
    # The PyPI resolver in Modal previously installed transformers==4.57.2,
    # which did not recognize that model_type. Install the newest Transformers
    # code after Unsloth so the runtime sees Gemma 4 support.
    .run_commands(
        "python -m pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo",
        "python -m pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/huggingface/transformers.git",
        "python - <<'PY'\nimport transformers, sys\nprint('Transformers runtime version:', transformers.__version__)\nPY",
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("gemma4tunnel-zotero-volume", create_if_missing=True)

# Secrets are injected from GitHub Actions environment variables. This avoids
# needing a local terminal or pre-created Modal dashboard secret.
secrets = [
    modal.Secret.from_dict(
        {
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "HF_USERNAME": os.environ.get("HF_USERNAME", ""),
            "ZOTERO_API_KEY": os.environ.get("ZOTERO_API_KEY", ""),
            "ZOTERO_LIBRARY_TYPE": os.environ.get("ZOTERO_LIBRARY_TYPE", "user"),
            "ZOTERO_LIBRARY_ID": os.environ.get("ZOTERO_LIBRARY_ID", ""),
            "ZOTERO_COLLECTION_KEY": os.environ.get("ZOTERO_COLLECTION_KEY", ""),
            "UNPAYWALL_EMAIL": os.environ.get("UNPAYWALL_EMAIL", ""),
        }
    )
]


@dataclass
class TriageDecision:
    item_key: str
    title: str
    doi: str
    license: str
    oa_status: str
    decision: str
    reason: str
    source_url: str


def _headers() -> dict[str, str]:
    key = os.environ.get("ZOTERO_API_KEY", "")
    h = {"Zotero-API-Version": "3"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _zotero_prefix() -> str:
    library_type = os.environ.get("ZOTERO_LIBRARY_TYPE", "user").strip().lower()
    library_id = os.environ.get("ZOTERO_LIBRARY_ID", "").strip()
    if not library_id:
        raise RuntimeError("Missing ZOTERO_LIBRARY_ID secret")
    if library_type not in {"user", "group"}:
        raise RuntimeError("ZOTERO_LIBRARY_TYPE must be 'user' or 'group'")
    plural = "users" if library_type == "user" else "groups"
    return f"https://api.zotero.org/{plural}/{library_id}"


def _request_json(url: str, *, headers: dict[str, str] | None = None) -> Any:
    import requests

    r = requests.get(url, headers=headers or {}, timeout=60)
    r.raise_for_status()
    return r.json()


def _paginate(url: str) -> list[dict[str, Any]]:
    import requests

    items: list[dict[str, Any]] = []
    headers = _headers()
    start = 0
    limit = 100
    while True:
        sep = "&" if "?" in url else "?"
        page_url = f"{url}{sep}limit={limit}&start={start}&format=json"
        r = requests.get(page_url, headers=headers, timeout=60)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        items.extend(chunk)
        if len(chunk) < limit:
            break
        start += limit
    return items


def _list_candidate_items() -> list[dict[str, Any]]:
    prefix = _zotero_prefix()
    collection_key = os.environ.get("ZOTERO_COLLECTION_KEY", "").strip()
    if collection_key:
        url = f"{prefix}/collections/{collection_key}/items/top"
    else:
        url = f"{prefix}/items/top"
    return _paginate(url)


def _children_for_item(item_key: str) -> list[dict[str, Any]]:
    return _paginate(f"{_zotero_prefix()}/items/{item_key}/children")


def _download_attachment(attachment_key: str, filename: str) -> Path | None:
    import requests

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename or f"{attachment_key}.pdf")
    out = PDF_DIR / f"{attachment_key}_{safe_name}"
    if out.exists() and out.stat().st_size > 1024:
        return out
    url = f"{_zotero_prefix()}/items/{attachment_key}/file"
    r = requests.get(url, headers=_headers(), timeout=180)
    if r.status_code >= 400:
        return None
    content_type = r.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not (filename or "").lower().endswith(".pdf"):
        return None
    out.write_bytes(r.content)
    return out


def _doi_from_item(data: dict[str, Any]) -> str:
    doi = (data.get("DOI") or data.get("doi") or "").strip()
    if doi:
        return doi.lower().replace("https://doi.org/", "").replace("doi:", "").strip()
    url = data.get("url") or ""
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", url, flags=re.I)
    return m.group(0).lower() if m else ""


def _tags(item: dict[str, Any]) -> set[str]:
    return {str(t.get("tag", "")).strip().lower() for t in item.get("data", {}).get("tags", [])}


def _crossref_license(doi: str) -> list[str]:
    if not doi:
        return []
    try:
        url = f"https://api.crossref.org/works/{doi}"
        msg = _request_json(url).get("message", {})
        return [str(x.get("URL", "")) for x in msg.get("license", []) if x.get("URL")]
    except Exception:
        return []


def _unpaywall_info(doi: str) -> dict[str, Any]:
    if not doi:
        return {}
    email = os.environ.get("UNPAYWALL_EMAIL", "").strip() or "research@example.org"
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        payload = _request_json(url)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a dict for optional JSON objects that may be null.

    Unpaywall can return JSON fields such as best_oa_location as null.
    Calling .get() on None crashes the ingest job, so all optional nested
    metadata must be normalized before access.
    """
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return a list for optional JSON arrays that may be null."""
    return value if isinstance(value, list) else []


def _normalize_license(raw_values: list[str]) -> str:
    joined = " ".join(x.lower() for x in raw_values if x)
    joined = joined.replace("_", "-").replace("/licenses/", "/licenses/")
    if not joined:
        return "unknown"
    if "creativecommons.org/publicdomain/zero" in joined or "cc0" in joined:
        return "cc0"
    if "publicdomain" in joined or "public-domain" in joined:
        return "public-domain"
    # Check ND and NC before more permissive forms.
    if "by-nc-nd" in joined:
        return "cc-by-nc-nd"
    if "by-nc-sa" in joined:
        return "cc-by-nc-sa"
    if "by-nc" in joined:
        return "cc-by-nc"
    if "by-nd" in joined:
        return "cc-by-nd"
    if "by-sa" in joined:
        return "cc-by-sa"
    if "creativecommons.org/licenses/by/" in joined or "cc-by" in joined:
        return "cc-by"
    if "all-rights-reserved" in joined:
        return "all-rights-reserved"
    return "custom-or-unknown"


def _triage_item(item: dict[str, Any]) -> TriageDecision:
    data = item.get("data", {})
    title = data.get("title") or "Untitled"
    item_key = item.get("key") or data.get("key") or ""
    doi = _doi_from_item(data)
    tags = _tags(item)

    # User-controlled tags are intentionally explicit and auditable.
    if "gemma4tunnel-own-work" in tags:
        return TriageDecision(item_key, title, doi, "own-work", "", "allow_public_training", "Zotero tag says this is the user's own work", data.get("url", ""))
    if "gemma4tunnel-allow-public-training" in tags:
        return TriageDecision(item_key, title, doi, "manual-allow", "", "allow_public_training", "Manual Zotero allow tag; verify permission evidence", data.get("url", ""))
    if "gemma4tunnel-rag-only" in tags:
        return TriageDecision(item_key, title, doi, "manual-rag-only", "", "rag_only", "Manual Zotero RAG-only tag", data.get("url", ""))
    if "gemma4tunnel-deny" in tags:
        return TriageDecision(item_key, title, doi, "manual-deny", "", "exclude", "Manual Zotero deny tag", data.get("url", ""))

    cr_licenses = _crossref_license(doi)
    upw = _as_dict(_unpaywall_info(doi))
    upw_licenses: list[str] = []
    oa_status = str(upw.get("oa_status") or "")
    source_url = data.get("url") or str(upw.get("doi_url") or "")

    for loc_raw in _as_list(upw.get("oa_locations")):
        loc = _as_dict(loc_raw)
        if loc.get("license"):
            upw_licenses.append(str(loc.get("license")))
        if loc.get("url_for_pdf") and not source_url:
            source_url = str(loc.get("url_for_pdf"))

    best_loc = _as_dict(upw.get("best_oa_location"))
    if best_loc.get("license"):
        upw_licenses.append(str(best_loc.get("license")))
    if best_loc.get("url_for_pdf"):
        source_url = str(best_loc.get("url_for_pdf"))

    license_name = _normalize_license(cr_licenses + upw_licenses)

    if license_name in {"cc0", "public-domain", "cc-by", "own-work", "manual-allow"}:
        return TriageDecision(item_key, title, doi, license_name, oa_status, "allow_public_training", "Permissive or explicit reuse license", source_url)
    if license_name == "cc-by-sa":
        return TriageDecision(item_key, title, doi, license_name, oa_status, "allow_public_training", "Allowed, but ShareAlike obligations must be considered in dataset/model card", source_url)
    if license_name in {"cc-by-nc", "cc-by-nc-sa"}:
        return TriageDecision(item_key, title, doi, license_name, oa_status, "rag_only", "NonCommercial license; keep out of broad public reusable model by default", source_url)
    if license_name in {"cc-by-nd", "cc-by-nc-nd"}:
        return TriageDecision(item_key, title, doi, license_name, oa_status, "exclude", "NoDerivatives license; do not use as training data by default", source_url)
    if oa_status in {"bronze", "green"}:
        return TriageDecision(item_key, title, doi, license_name, oa_status, "rag_only", "Free-to-read/OA copy found but no clear reuse license", source_url)
    if not doi:
        return TriageDecision(item_key, title, doi, license_name, oa_status, "rag_only", "No DOI/license evidence; use private RAG only", source_url)
    return TriageDecision(item_key, title, doi, license_name, oa_status, "rag_only", "License unclear or not permissive enough for public model training", source_url)


def _extract_pdf_text(path: Path) -> str:
    import fitz  # PyMuPDF

    chunks: list[str] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                chunks.append(f"[page {i + 1}] {text}")
    return "\n".join(chunks)


def _chunk_text(text: str, max_chars: int = 4500) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    paras = re.split(r"(?<=\.)\s+(?=[A-Z0-9])", text)
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf} {para}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = para[:max_chars]
    if buf:
        chunks.append(buf)
    return chunks


def _make_record(decision: TriageDecision, chunk: str, chunk_id: int) -> dict[str, Any]:
    # This creates SFT records that teach source-grounded technical phrasing.
    # For high-quality assistant behavior, later add human/LLM-generated Q&A and
    # design-review examples based only on the allowed corpus.
    source_line = f"Source: {decision.title}; DOI: {decision.doi or 'none'}; license: {decision.license}."
    messages = [
        {
            "role": "system",
            "content": "You are Gemma4Tunnel, a careful tunnel boring machine and tunnelling engineering assistant. Use engineering terminology precisely, state assumptions, and avoid inventing source claims.",
        },
        {
            "role": "user",
            "content": "Extract reusable tunnelling/TBM engineering knowledge from this allowed source excerpt. Preserve important parameters, mechanisms, risks, and caveats. Do not add unsupported claims.\n\n" + source_line,
        },
        {
            "role": "assistant",
            "content": chunk,
        },
    ]
    return {
        "id": hashlib.sha256(f"{decision.item_key}-{chunk_id}".encode()).hexdigest()[:16],
        "messages": messages,
        "source_title": decision.title,
        "source_doi": decision.doi,
        "source_url": decision.source_url,
        "source_license": decision.license,
        "zotero_item_key": decision.item_key,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_audit(path: Path, decisions: list[TriageDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["item_key", "title", "doi", "license", "oa_status", "decision", "reason", "source_url"],
        )
        writer.writeheader()
        for d in decisions:
            writer.writerow(d.__dict__)


def _dataset_card(dataset_repo: str, audit_counts: dict[str, int]) -> str:
    return f"""---
language:
- en
license: other
pretty_name: Gemma4Tunnel Open Tunnelling Corpus
tags:
- tunnelling
- tunnel-boring-machine
- tbm
- geotechnical-engineering
- civil-engineering
- text-generation
---

# Gemma4Tunnel Open Tunnelling Corpus

This dataset was generated from a Zotero library using a conservative license triage process. Source PDFs are not included in this repository.

## Intended use

Fine-tuning and evaluating Gemma4Tunnel, a tunnelling and tunnel boring machine assistant.

## License policy

Only items classified as `allow_public_training` are included in `train.jsonl` and `validation.jsonl`.

Audit counts:

```json
{json.dumps(audit_counts, indent=2)}
```

Review `license_audit.csv` before publishing a model trained on this data.

## Caveat

This is not legal advice. Ambiguous sources are routed to RAG-only or excluded by default.
"""


def _model_card(model_repo: str, base_model: str, dataset_repo: str) -> str:
    return f"""---
language:
- en
license: other
base_model: {base_model}
tags:
- gemma
- unsloth
- lora
- tunnelling
- tunnel-boring-machine
- geotechnical-engineering
- civil-engineering
pipeline_tag: text-generation
---

# Gemma4Tunnel

Gemma4Tunnel is a LoRA adapter fine-tuned from `{base_model}` for tunnel boring machine and tunnelling engineering research assistance.

## Dataset

Training data: `{dataset_repo}`.

The data pipeline is designed to include only sources classified as suitable for public model training. Review the dataset repo's `license_audit.csv` for source-level decisions.

## Intended use

- Research and education in tunnelling/TBM engineering
- Drafting technical explanations
- Preparing checklists, risk registers, and study notes

## Limitations

- Not a substitute for professional engineering judgement.
- Must not be used as the only basis for design, construction, safety-critical, legal, or contractual decisions.
- May still hallucinate; use RAG with source citations for paper-specific answers.

## Recommended use

Use this adapter together with a private RAG system over your Zotero library for exact source-grounded answers.
"""


@app.function(
    image=base_image,
    secrets=secrets,
    volumes={str(VOLUME_PATH): volume},
    timeout=60 * 60 * 6,
)
def ingest_zotero_to_hf(dataset_repo: str, max_chars: int = 4500) -> dict[str, Any]:
    from huggingface_hub import HfApi, create_repo

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = _list_candidate_items()
    decisions: list[TriageDecision] = []
    records: list[dict[str, Any]] = []

    for item in items:
        item_key = item.get("key") or item.get("data", {}).get("key")
        if not item_key:
            continue
        decision = _triage_item(item)
        decisions.append(decision)
        if decision.decision != "allow_public_training":
            continue

        for child in _children_for_item(item_key):
            cdata = child.get("data", {})
            if cdata.get("itemType") != "attachment":
                continue
            content_type = str(cdata.get("contentType", "")).lower()
            filename = str(cdata.get("filename", ""))
            if "pdf" not in content_type and not filename.lower().endswith(".pdf"):
                continue
            pdf_path = _download_attachment(child.get("key") or cdata.get("key"), filename)
            if not pdf_path:
                continue
            try:
                text = _extract_pdf_text(pdf_path)
            except Exception as exc:
                decisions.append(
                    TriageDecision(
                        decision.item_key,
                        decision.title,
                        decision.doi,
                        decision.license,
                        decision.oa_status,
                        "extract_failed",
                        f"PDF extraction failed: {exc}",
                        decision.source_url,
                    )
                )
                continue
            for idx, chunk in enumerate(_chunk_text(text, max_chars=max_chars)):
                if len(chunk) >= 300:
                    records.append(_make_record(decision, chunk, idx))

    random.Random(42).shuffle(records)
    n_val = max(1, int(0.05 * len(records))) if records else 0
    val = records[:n_val]
    train = records[n_val:]

    audit_path = OUT_DIR / "license_audit.csv"
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "validation.jsonl"
    readme_path = OUT_DIR / "README.md"

    _write_audit(audit_path, decisions)
    _write_jsonl(train_path, train)
    _write_jsonl(val_path, val)

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.decision] = counts.get(d.decision, 0) + 1
    readme_path.write_text(_dataset_card(dataset_repo, counts), encoding="utf-8")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN secret")

    create_repo(dataset_repo, repo_type="dataset", private=False, exist_ok=True, token=hf_token)
    api = HfApi(token=hf_token)
    for path in [audit_path, train_path, val_path, readme_path]:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=dataset_repo,
            repo_type="dataset",
        )

    volume.commit()
    return {
        "candidate_items": len(items),
        "records": len(records),
        "train_records": len(train),
        "validation_records": len(val),
        "audit_counts": counts,
        "dataset_repo": dataset_repo,
    }


def _format_dataset(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


@app.function(
    image=train_image,
    secrets=secrets,
    volumes={str(VOLUME_PATH): volume},
    gpu=MODAL_GPU,
    timeout=60 * 60 * 24,
)
def train_lora_on_modal(
    dataset_repo: str,
    model_repo: str,
    base_model: str = "google/gemma-4-12B-it",
    max_steps: int = 20,
    max_seq_length: int = 4096,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict[str, Any]:
    # Unsloth must be imported before TRL / Transformers / PEFT so its patches are active.
    import unsloth  # noqa: F401
    from unsloth import FastModel
    import torch
    import transformers
    from datasets import load_dataset
    from huggingface_hub import HfApi, create_repo
    from trl import SFTConfig, SFTTrainer

    print(f"Transformers runtime version inside train_lora_on_modal: {transformers.__version__}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing HF_TOKEN secret")

    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
        token=hf_token,
    )

    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = load_dataset(dataset_repo, data_files={"train": "train.jsonl", "validation": "validation.jsonl"})
    train_ds = ds["train"].map(lambda x: _format_dataset(x, tokenizer), remove_columns=ds["train"].column_names)
    eval_ds = ds["validation"].map(lambda x: _format_dataset(x, tokenizer), remove_columns=ds["validation"].column_names)

    output_dir = VOLUME_PATH / "runs" / "gemma4tunnel_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=5,
        save_steps=max(25, max_steps // 4),
        eval_strategy="steps" if len(eval_ds) else "no",
        eval_steps=max(25, max_steps // 4),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        args=args,
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    readme = _model_card(model_repo, base_model, dataset_repo)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    create_repo(model_repo, repo_type="model", private=False, exist_ok=True, token=hf_token)
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=model_repo,
        repo_type="model",
        commit_message="Upload Gemma4Tunnel LoRA adapter",
    )

    volume.commit()
    return {
        "model_repo": model_repo,
        "dataset_repo": dataset_repo,
        "base_model": base_model,
        "max_steps": max_steps,
        "max_seq_length": max_seq_length,
        "gpu": MODAL_GPU,
    }


@app.local_entrypoint()
def main(
    action: str = "ingest",
    dataset_repo: str = "",
    model_repo: str = "",
    base_model: str = "google/gemma-4-12B-it",
    max_steps: int = 20,
    max_seq_length: int = 4096,
    gpu: str = "H100",
):
    # The gpu argument is accepted for GitHub Actions UI clarity. The actual GPU
    # is set by MODAL_GPU at import time.
    if not dataset_repo:
        hf_user = os.environ.get("HF_USERNAME", "").strip()
        if not hf_user:
            raise RuntimeError("Provide dataset_repo or set HF_USERNAME")
        dataset_repo = f"{hf_user}/gemma4tunnel-data"
    if not model_repo:
        hf_user = os.environ.get("HF_USERNAME", "").strip()
        if not hf_user:
            raise RuntimeError("Provide model_repo or set HF_USERNAME")
        model_repo = f"{hf_user}/gemma4tunnel"

    action = action.strip().lower()
    if action not in {"ingest", "train", "all"}:
        raise RuntimeError("action must be ingest, train, or all")

    if action in {"ingest", "all"}:
        result = ingest_zotero_to_hf.remote(dataset_repo=dataset_repo)
        print(json.dumps(result, indent=2))

    if action in {"train", "all"}:
        result = train_lora_on_modal.remote(
            dataset_repo=dataset_repo,
            model_repo=model_repo,
            base_model=base_model,
            max_steps=max_steps,
            max_seq_length=max_seq_length,
        )
        print(json.dumps(result, indent=2))

# Contract-Preserving Semantic Extraction (CPSE)

This repository contains the CPSE implementation, prompts, structural schema, and sanitized configurations for low-resource scientific-PDF extraction.

## Release boundary

This source release deliberately excludes PDFs, Gold annotations, the task-specific schema, cached model responses, run outputs, experimental results, credentials, private service URLs, and proxy settings. End-to-end reproduction therefore requires an independently authorized PDF--Gold dataset and compatible schema.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
Copy-Item cpse\textgrad_validation\config.example.yaml config.yaml
```

Set `MODEL_API_KEY` in `.env`, then edit `config.yaml` to select the extraction and evaluation models supported by your API provider.

## Dataset layout

Place authorized document pairs under `cpse/data/` using matching file stems:

```text
cpse/data/
  paper_001.pdf
  paper_001.json
  paper_002.pdf
  paper_002.json
  schema.json
```

The JSON files must conform to the schema at `cpse/data/schema.json`; update the corresponding `schema_path` in `config.yaml` if you store it elsewhere.

## Run CPSE

First validate the local configuration and dataset without making model calls:

```powershell
python -m cpse.textgrad_validation --config config.yaml --run-id cpse-reproduction-01 --preflight
```

Then run the primary two-stage calibration workflow:

```powershell
python -m cpse.textgrad_validation --config config.yaml --run-id cpse-reproduction-01
```

The implementation package is `cpse/textgrad_validation/`. Its configuration files document optional baseline, evaluation, and analysis modes.

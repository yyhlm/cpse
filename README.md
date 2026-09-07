# Source release

This repository contains the implementation, prompts, schema, sanitized experiment configurations, and offline tests for low-resource scientific PDF extraction with TextGrad. It deliberately excludes copyrighted PDFs, Gold annotations, run results, caches, credentials, private service URLs, and proxy settings.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `cpse/textgrad_validation/config.example.yaml`, replace the model and endpoint placeholders, and set `MODEL_API_KEY` using `.env.example` as a template. Place an authorized PDF/Gold dataset under `cpse/data/`, then run:

```bash
python -m cpse.textgrad_validation --config <config.yaml> --run-id <run-id>
python -m pytest -q cpse/tests
```

The primary two-stage configuration is `cpse/textgrad_validation/config_two_stage2.yaml`. Detailed modes, artifact layouts, recovery commands, and ablation protocols are documented in `cpse/README.md` and `cpse/textgrad_validation/README.md`.

## Reproduction boundary

Published result verification should use the paired public artifact bundle containing frozen predictions and evaluation artifacts. This source release alone cannot reproduce PDF-dependent scores without authorized source documents and Gold annotations.

The schema and prompts are included because they define the evaluated protocol. Raw source documents are excluded because redistribution rights remain with the respective publishers.

## License

No open-source license is selected automatically. Add the intended license before making the GitHub repository public.

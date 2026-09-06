# Source release: TextGrad scientific PDF extraction experiment

This package contains the full `cpse/` experiment source: implementation, prompts, sanitized configuration templates, schema, and user documentation. It deliberately excludes PDFs, Gold annotations, run results, caches, environment files, credentials, service URLs, and proxy settings.

## Reproduction boundary

Install `requirements.txt`, place an authorized PDF/Gold dataset under `cpse/data/`, configure your own endpoint and credentials locally, then follow `cpse/README.md`. Published result verification should use the paired public artifact bundle, which contains frozen predictions and evaluation artifacts.

The schema and prompts are included because they define the evaluated protocol. 

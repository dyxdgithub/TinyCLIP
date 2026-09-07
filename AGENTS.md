# TinyCLIP Agent Guide

This file is the navigation map for AI agents working in this repository.
Details live in the linked documents, not here.

## 1 Quick Start

- [Architecture](docs/ARCHITECTURE.md) - package layers, dependency rules, data flow.
- [Development Setup](docs/DEVELOPMENT.md) - install, test, lint, inference, training.
- [Quality Standards](docs/QUALITY.md) - linters and code invariants.
- [Existing README](README.md) - paper links, model zoo, high-level usage.

## 2 Architecture

| Section | Document | Description |
|---------|----------|-------------|
| 2.1 | [System Architecture](docs/ARCHITECTURE.md) | Layer hierarchy, Mermaid diagrams, forbidden dependencies. |
| 2.2 | [Model Construction](docs/design-docs/model-construction.md) | `open_clip` config loading, encoders, pretrained weights. |
| 2.3 | [Distillation And Pruning](docs/design-docs/distillation-and-pruning.md) | Soft loss, L0 masks, weight inheritance, model pruning. |
| 2.4 | [Training Pipeline](docs/design-docs/training-pipeline.md) | Data loaders, distributed setup, train/eval loop, zero-shot eval. |
| 2.5 | [Design Doc Index](docs/design-docs/index.md) | Component-level documentation map. |

## 3 Runtime Contract

| Section | Document | Description |
|---------|----------|-------------|
| 3.1 | [Environment Contract](harness/config/environment.json) | Runtime, startup, environment variables, functional scenarios. |
| 3.2 | [Setup Script](harness/scripts/setup-env.sh) | Idempotent environment setup. |
| 3.3 | [Start Script](harness/scripts/start-server.sh) | Startup smoke check. |
| 3.4 | [Teardown Script](harness/scripts/teardown-env.sh) | Cleanup. |

## 4 Quality And Standards

| Section | Document | Description |
|---------|----------|-------------|
| 4.1 | [Quality Standards](docs/QUALITY.md) | Enforced and recommended invariants. |
| 4.2 | [Tech Debt Tracker](docs/exec-plans/tech-debt-tracker.md) | Known warnings and remediation paths. |
| 4.3 | [Architecture Linter](scripts/lint_deps.py) | Layer dependency enforcement. |
| 4.4 | [Quality Linter](scripts/lint_quality.py) | File size, secrets, debug statement checks. |

## 5 Development Commands

Run from the repository root.

```bash
python -m pip install -r requirements-training.txt
python -m pip install -e .

make lint-arch    # dependency + quality linters
make build        # syntax compile all source files
make test         # pytest (tests directory must exist)
```

On PowerShell, set `PYTHONPATH` before training/evaluation:

```powershell
$env:PYTHONPATH = "src"
python -m training.main_for_test --help
```

See [Development Setup](docs/DEVELOPMENT.md) for the full command reference.

## 6 Key Directories

| Directory | Layer | Purpose |
|-----------|-------|---------|
| `src/open_clip/` | L0 | CLIP model library, config registry, pretrained weights, pruning primitives. |
| `src/data/` | L1 | Standalone dataset acquisition/export tooling. |
| `src/training/` | L2 | Distributed training, data pipelines, eval, logging, scheduling. |
| `my_code/` | L3 | Local experiments: finetune, dataset previews, ImageNet-200 tests. |
| `inference.py`, `measure_throughput.py` | L4 | Top-level library entry points. |
| `docs/` | - | Architecture, development, quality, and design docs. |
| `scripts/` | - | Agent linters. |
| `harness/` | - | Runtime contract and environment scripts. |

## 7 Working Rules

- Before executing a task, read `my_code/AGENTS.md`.
- `open_clip` is the reusable core. It must not import `training`, `data`, `my_code`, or top-level scripts.
- `training` may import `open_clip` and its own submodules only.
- `my_code` is experimental. It may import `open_clip` and `training`, but nothing may import it.
- Keep checkpoint/model/download paths outside version control.
- Before editing, run `make lint-arch` to verify the dependency graph has not been violated.

## 8 Data And Model Artifacts

- Model architecture files live in `src/open_clip/model_configs/*.json`.
- Pretrained checkpoint URLs live in `src/open_clip/pretrained.py`.
- Local checkpoints and datasets are not part of the harness contract.
- Do not commit model weights, ImageNet, LAION, or YFCC data.

## 9 Known Technical Debt

- A committed Hugging Face token is present at `my_code/test/dataset.py:10`.
- `src/open_clip/model.py` exceeds the recommended 1000-line limit.
- The Makefile `test` target points at a missing `tests/` directory.

See [Tech Debt Tracker](docs/exec-plans/tech-debt-tracker.md).

## 10 Change Protocol

1. Read [Architecture](docs/ARCHITECTURE.md) before moving code across layers.
2. Make the smallest module-level change that satisfies the task.
3. Run `make build` and `make lint-arch`.
4. Add focused pytest coverage when changing `open_clip` behavior.
5. Update the relevant design doc when a public interface or data flow changes.
6. Keep new dependencies in the same `requirements*.txt` tier as their consumer.
7. Prefer small, dependency-free helpers in `open_clip` over trainer-specific logic.

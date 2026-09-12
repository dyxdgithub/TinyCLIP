# my_code Agent Guide

This file scopes agent work inside `my_code/`. The repository-root
[`AGENTS.md`](../AGENTS.md) remains canonical for package layers, build
commands, and project-wide rules. This file adds experiment-area constraints.

## 1 Scope

- `my_code/` is the local experiment area for TinyCLIP.
- It is layer L3 in the root architecture: it may import `open_clip` and
  `training`, but no production package may import it.
- Do not move reusable model or trainer logic into `my_code/`. If code must be
  shared, promote it to `src/open_clip/` or `src/training/` only with a task
  that explicitly asks for that.
- Do not modify unrelated repository areas while working on a `my_code` task.

## 2 Current Structure

| Directory / File | Purpose |
|------------------|---------|
| `my_code/finetune/finetune.py` | Full or partial TinyCLIP fine-tuning entry point. |
| `my_code/test/test_image200.py` | ImageNet-200 zero-shot classification and retrieval benchmark. |
| `my_code/test/dataset.py` | Hugging Face dataset preview tool. |
| `my_code/data/augment.py` | FFT-based hard-negative image augmentation. |
| `my_code/data/clear.py` | FiftyOne dataset cleanup utility. |
| `my_code/data/ImageNet-200/` | Local ImageNet-200 images and label maps. |
| `my_code/data/OpenImage/` | OpenImage dataset preprocessing, metadata, and download tools. |

## 3 Entry Points

Run from the repository root with `src/` on `PYTHONPATH`.

```powershell
$env:PYTHONPATH = "src"
```

| Command | Purpose |
|---------|---------|
| `python my_code/test/test_image200.py` | ImageNet-200 zero-shot benchmark. |
| `python my_code/test/dataset.py --dataset wikimedia/wit_base --num-samples 5` | Dataset preview. |
| `python my_code/finetune/finetune.py --help` | Show fine-tuning arguments. |
| `python my_code/data/augment.py` | Run the augmentation demo/utility. |
| `python my_code/data/clear.py` | List or delete FiftyOne datasets. |

> Sources: [`my_code/test/test_image200.py:28-44`](),
> [`my_code/test/dataset.py:16-62`](),
> [`my_code/finetune/finetune.py:41-172`]().

## 4 Dependency Rules

- `my_code` scripts may import `open_clip` and `training`.
- `my_code` submodules may import one another.
- `open_clip`, `training`, and `src/data` must not import `my_code`.
- Keep experimental scripts self-contained; do not add a package `__init__.py`
  that causes `my_code` to be imported as an installed library.

> Enforced by root [`scripts/lint_deps.py`](../scripts/lint_deps.py).

## 5 Data And Secrets

- Local datasets live under `my_code/data/`.
- All tasks involving models default to running on the GPU. Use CPU only when the task explicitly requires it or GPU execution is unavailable.
- For file-operation tasks under `C:\Users\ASUS\Desktop\大模型代码\TinyCLIP\my_code\data\OpenImage\meta`, treat `meta` as the script root directory. Resolve default input, output, and generated-file paths relative to `meta` unless the task explicitly specifies another location.
- Place data-processing code under `my_code/data/OpenImage/meta/scripts/`; do not add data-processing scripts directly in `meta`.
- For every code implementation task, Codex may run focused verification such as syntax checks, unit tests, or synthetic small-data tests. Do not run the final end-to-end code, model jobs, or full-dataset data-processing jobs; provide the user with the exact commands to run those manually instead.
- Long-running tasks that download, convert, or otherwise process a long list must display ongoing progress, persist progress, and support resuming after interruption without repeating completed items. Progress output must include completed work and either a total count or a clearly stated indeterminate total.
- Every runtime parameter that has configurable or optional values must document its accepted values, defaults, and behavior in the script's `--help` output.
- Do not execute download tasks yourself while acting in Codex. For every download task, provide the required workflow and commands so the user can run the download manually. Scripts may download their declared runtime dependencies or data when the user runs them.
- Do not commit model weights, dataset images, downloaded archives, or cache
  files.
- Never hardcode Hugging Face or API tokens in source.
- `my_code/test/dataset.py:10` currently hardcodes `HF_TOKEN`. Replace it with
  `os.getenv("HF_TOKEN")` and rotate the leaked value before further use.

## 6 Task Type And Commit Separation

Every `my_code` task must be one of:

### 6.1 Planning / Log-Only Task

Allowed:

```text
my_code/AGENTS.md
my_code/TASK_LOG.md
my_code/task_log/**/*.md
```

Forbidden:

```text
*.py
my_code/data/**
```

Purpose: create or revise task specs, rules, indexes, or planning docs.

### 6.2 Code Implementation Task

Allowed:

```text
Only code and data files explicitly listed by the active task detail.
```

Forbidden:

```text
my_code/AGENTS.md
my_code/TASK_LOG.md
my_code/task_log/**/*.md
```

Purpose: implement exactly one task. Do not update logs in the code commit.

### 6.3 Completion-Log Task

Allowed:

```text
my_code/TASK_LOG.md
my_code/task_log/**/*.md
```

Forbidden:

```text
Code, tests, scripts, configs, datasets, checkpoints.
```

Purpose: mark a task DONE and record the implementation commit, changed files,
tests, and exact task boundaries after code passes.

## 7 Test Policy

- Codex may run focused syntax checks, linters, unit tests, and synthetic or
  small-data entry-point tests while implementing code.
- Do not run final end-to-end entry points, model jobs, or full-dataset
  data-processing jobs in Codex. Provide the exact commands for the user to
  run manually.
- Do not make real paid API calls in Codex or user-run tests. Dataset previews
  and fine-tuning may use Hugging Face only when the task explicitly requires it.
- On Windows, Codex or users may keep bytecode out of the worktree:

```powershell
$env:PYTHONPYCACHEPREFIX = '.local_test_tmp\pycache'
```

## 8 Notation And Style

- Use ASCII notation in task logs, prompts, and new comments.
- Keep module-level comments in English unless the surrounding file already
  uses Chinese comments.
- Prefer stable identifiers and paths over prose-only descriptions.

## 9 Git Policy

- One user task should produce one clearly reversible commit.
- Planning/log-only commits, code commits, and completion-log commits must be
  separate.
- Stage only files touched for the current task.
- Do not use `git reset --hard`, force push, or history rewriting for rollback.
  Prefer `git revert` for pushed work unless explicitly instructed otherwise.

## 10 Quick Checks

```bash
make build
make lint-arch
python scripts/lint_deps.py
python scripts/lint_quality.py
```

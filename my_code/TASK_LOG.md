# TASK_LOG.md

This file is the task index for the `my_code/` TinyCLIP experiment area.
Detailed task specs live under `my_code/task_log/planned/` once created.

## 1 Workflow Rules

- Planning/log-only commits must be separate from code implementation commits.
- Code implementation commits must not edit `TASK_LOG.md` or
  `my_code/task_log/**/*.md`.
- Completion-log commits must be separate from code commits.
- Use ASCII notation in task specs, prompts, and comments.
- If a task type is ambiguous, stop and ask for clarification.

## 2 Status Values

| Status | Meaning |
|--------|---------|
| `TODO` | Task is planned but not started. |
| `IN_PROGRESS` | Active task is being implemented or planned. |
| `DONE` | Code/tests passed and completion log is recorded. |
| `BLOCKED` | Work is blocked by an external dependency or missing information. |

## 3 Historical Archive

| Range | Detail |
|---|---|
| None | No historical tasks have been archived yet. |

## 4 Active Task Index

| ID | Title | Status | Detail | Implementation Commit | Completion Log Commit |
|---|---|---|---|---|---|
| M001 | Embed class names and export cosine similarities | IN_PROGRESS | [planned/M001_class_name_embedding_similarity.md](task_log/planned/M001_class_name_embedding_similarity.md) | - | - |

The active task is planned in the linked detail file. Implementation and
completion-log commits must remain separate.

## 5 Task Detail File Convention

Use one file per task under `my_code/task_log/planned/`, for example:

```text
my_code/task_log/planned/M001_dataset_token_fix.md
```

Each detail file must record:

- Task ID, title, and task type.
- Allowed files and forbidden files.
- Current behavior with source references.
- Required changes and acceptance criteria.
- Test commands and expected results.

## 6 Completion Log Format

After code/tests pass, record the result in a separate completion-log commit:

```text
ID: M001
Title: Move HF token to environment
Status: DONE
Implementation commit: <hash>
Completion log commit: <hash>
Changed files:
  - my_code/test/dataset.py
Tests:
  - make build
  - make lint-arch
  - python my_code/test/dataset.py --dataset wikimedia/wit_base --num-samples 1
Boundary:
  - Removed hardcoded token and read HF_TOKEN from the environment.
  - No changes outside my_code/test/dataset.py.
```

## 7 Reference

The workflow is adapted from the Kenan/KAN collaboration rules. The root
[`AGENTS.md`](../AGENTS.md) continues to govern package boundaries and
project-wide commands.


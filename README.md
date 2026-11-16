# SmolLangForce

CMPE 257 final project — Pratik Bhandari (SJSU)

Reproducing [LangForce](https://arxiv.org/abs/2601.15197) on a smaller model: SmolVLA (~605M) instead of the paper's 4B QwenGR00T, on the LIBERO benchmark.

## Phase 1: Does SmolVLA ignore language?

For 200 frames per LIBERO suite, we compute SmolVLA's loss with the **true**, **blank**, and **swapped** instruction (same images and noise, only the text changes).
Swap gap = loss_swap − loss_true. Small gap = language ignored (vision shortcut).

| Suite | Swap gap (95% CI) |
|---|---|
| Goal | 0.713 [0.612, 0.826] |
| Spatial | 0.294 [0.267, 0.324] |
| Long | 0.243 [0.203, 0.288] |
| Object | 0.085 [0.066, 0.105] |

Same ranking as the paper's Table 6 (Goal ≫ Spatial > Long > Object): SmolVLA shows the vision shortcut on Object and relies on language on Goal.

## Phase 2: LangForce on SmolVLA

In progress.

## How to run

1. Open the notebook in Google Colab (T4 GPU).
2. Run `!pip install -q "lerobot[smolvla,dataset]==0.7.0"`, then restart the session.
3. Mount Google Drive and run the cells in order.

Results are saved to `results/phase1_offline.csv`.

## Setup

- Model: `HuggingFaceVLA/smolvla_libero`
- Data: `lerobot/libero_{goal,object,spatial,10}_image`
- Library: LeRobot 0.7.0

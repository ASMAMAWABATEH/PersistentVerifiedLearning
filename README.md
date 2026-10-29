# VerityLoop

**Evidence-Driven AI Improvement Through Verified Learning and Knowledge Distillation**

VerityLoop is a research project investigating two approaches to improving AI systems through controlled, reproducible experiments:

1. **Verified iterative learning:** Generate candidate solutions, evaluate them using independent tests, curate verified examples, and assess whether retraining improves performance.
2. **Knowledge distillation:** Transfer predictive behavior from a teacher model to a smaller student model and compare its performance with a student trained without distillation.

The project emphasizes empirical evidence, independent verification, reproducibility, and transparent reporting of both successful and unsuccessful experiments.

## Research Objectives

- Implement a bounded iterative-learning pipeline with explicit acceptance criteria.
- Establish a reliable benchmark using independently testable programming tasks.
- Compare baseline and updated models on held-out data.
- Compare a label-supervised student against a student trained using knowledge distillation.
- Measure correctness, generalization, inference latency, and model size.
- Document failure cases, limitations, and experimental costs.

## Research Tracks

### Track A — Verified Iterative Learning

The verified-memory pipeline follows:

`Generate → Verify → Curate → Evaluate → Accept or Reject`

Only verified training-split examples may enter memory. A proposed memory update is accepted only when it satisfies the development-set gate; otherwise, the existing memory is retained. This changes prompting context, not model weights.

### Track B — Knowledge Distillation

The experiment compares:

- **Teacher:** The reference model that produces predictions.
- **Baseline student:** A smaller model trained using hard labels.
- **Distilled student:** A comparable student trained using hard-label supervision and teacher soft targets.

The teacher's predictions are training signals, not proof of correctness. All models must be evaluated against the same held-out examples.

The current NumPy MLP performs genuine distillation of three-class logits on synthetic two-feature data, using `L_KD = αT² KL(p_teacher || p_student) + (1 − α) CE(student, targets)`. It is not a token-level code model: verified coding examples are prompt/code pairs, and the repository has no tokenizer or trainable code-generation model to provide compatible token logits. Therefore the synthetic experiment does not train on verified coding examples and must not be reported as coding-model distillation. The verified-memory pipeline likewise does not fine-tune weights.

## Experimental Principles

- Fix and record random seeds.
- Separate training, development, and final-test data.
- Use development data for model and acceptance decisions.
- Keep final-test results out of training and model-selection decisions.
- Compare methods using consistent evaluation conditions.
- Report negative results and limitations honestly.
- Keep experiments feasible on CPU-only hardware.

## Project Structure

```text
VerityLoop/
├── configs/                 # Experiment configuration
├── data/
│   ├── raw/                 # Original benchmark data
│   └── processed/           # Prepared datasets and splits
├── experiments/             # Experiment entry points
├── results/
│   ├── checkpoints/         # Model checkpoints
│   ├── figures/             # Plots and visualizations
│   └── logs/                # Experiment logs
├── src/verityloop/
│   ├── benchmark/           # Tasks, test runner, dataset
│   ├── recursive_learning/  # Verification and iterative updates
│   ├── distillation/        # Teacher, student, and loss functions
│   ├── evaluation/          # Metrics and data splits
│   └── utils/               # Configuration and reproducibility
├── tests/                   # Automated tests
├── docs/                    # Experimental protocol
├── pyproject.toml
└── requirements.txt
```

## Development Status

The repository structure has been initialized. Benchmark construction, baseline implementation, and model experiments will be developed and validated incrementally.

No performance claims are made until supported by recorded experimental results.

## Reproducibility

Experiment settings, data splits, evaluation procedures, and results should be recorded so that each comparison can be reproduced and independently checked.

## Limitations

This project is an experimental research framework, not a general-purpose coding assistant. Its conclusions will be limited by the benchmark, model capacity, available compute, and evaluation protocol.

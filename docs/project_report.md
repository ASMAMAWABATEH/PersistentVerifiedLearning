# VerityLoop Project Report

**Status: 29 October 2026**

## Summary

VerityLoop is a CPU-friendly research prototype for testing controlled, evidence-based learning on programming tasks. It combines a verified-example memory pipeline with a small NumPy knowledge-distillation demonstration. Its central design principle is to distinguish verified evidence and measured model updates from changes to prompts or memory.

The project has working benchmark and evaluation components, a bounded acceptance-and-rollback flow, and a synthetic-data teacher–student experiment. It does **not** currently fine-tune or distill a code-generation model using the verified coding examples.

## Verified learning and safeguards

The benchmark contains independently testable coding tasks. Task families are separated into training, development, and final-test splits (60%, 20%, and 20%, using seed 42) to reduce leakage between related tasks. In the persistent-learning pipeline, generated candidates are independently tested and only fully verified candidates from the training split can be proposed for persistent memory.

The pipeline evaluates the proposed memory against the active memory on development tasks. It preserves the configured acceptance threshold (at least a 0.02 gain, with no development regression) and retains the prior active memory when a proposal is rejected. The persistent-learning runner does not evaluate the final-test split. Importantly, accepted memory changes affect examples supplied as prompting context; they do not update a model’s learned parameters.

The recorded persistent-learning run verified all 24 training candidates and proposed 24 memory entries. Development pass rate remained 1.00 for both active and proposed memory, so the required gain was not met. The proposal was rejected and rollback preserved the active memory. No model training or weight update occurred, and the final-test split remained untouched.

## Distillation implementation and evidence

The distillation package contains a one-hidden-layer NumPy MLP that exposes class logits and can be trained with SGD. Its training objective is:

`L_KD = αT² KL(p_teacher || p_student) + (1 − α) CE(student, targets)`

Here, `T` is the configurable temperature and `α` weights teacher soft-target loss. The loss helper and training gradient use this convention. The experiment’s teacher and student are compatible for three-class classification; it trains a 64-hidden-unit teacher and a 16-hidden-unit student on a small synthetic two-feature dataset.

The recorded synthetic experiment reported development accuracy of 0.927 for the teacher, 0.918 for the supervised student, and 0.927 for the distilled student. On its held-out synthetic test set, the corresponding scores were 0.933, 0.936, and 0.935. The student has 99 parameters compared with the teacher’s 387. These figures are results from the synthetic mechanism benchmark, not from VerityLoop coding tasks, and do not establish that distillation improves code generation.

Focused tests check the weighted loss against its definition and confirm that training changes the synthetic student’s parameters. The full project test suite passed: **24 tests passed**. Ruff lint checks for the modified Python files also passed.

## Limitation and next step

Verified coding examples are prompt/code pairs, while the available MLP accepts fixed-size numeric inputs and produces three-class logits. The repository has no tokenizer or trainable code-generation model that exposes compatible token-level logits for both teacher and student. Mapping code examples to synthetic class labels would not be genuine coding-model distillation, so the implementation does not do that. The synthetic training run and its parameter updates are separate from the verified-example pipeline; **no code model weights were updated**.

A genuine next step requires a suitable trainable code-generation student and a compatible teacher interface that returns token-level logits or probabilities over the same vocabulary. Any such training should consume only verified training-split examples, keep development data for evaluation and acceptance decisions, and preserve the final-test isolation and rollback policies.

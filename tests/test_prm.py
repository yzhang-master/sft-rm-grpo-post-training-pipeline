import pytest

from post_training.prm import (
    build_step_examples,
    candidate_steps,
    monte_carlo_step_label,
)


def test_monte_carlo_labels_measure_continuation_success(dataset):
    task = dataset.math_train[0]
    correct, wrong = candidate_steps(task, [])[:2]
    correct_label = monte_carlo_step_label(task, [], correct, 256, seed=8)
    assert 0.1 < correct_label < 0.3
    assert monte_carlo_step_label(task, [], wrong, 16, seed=8) == 0.0


def test_terminal_process_labels_are_exact_and_reproducible(dataset):
    task = dataset.math_train[0]
    first = candidate_steps(task, [])[0]
    correct, wrong = candidate_steps(task, [first])[:2]
    assert monte_carlo_step_label(task, [first], correct, 4, seed=2) == 1.0
    assert monte_carlo_step_label(task, [first], wrong, 4, seed=2) == 0.0
    assert monte_carlo_step_label(task, [], first, 8, seed=5) == monte_carlo_step_label(
        task, [], first, 8, seed=5
    )
    with pytest.raises(ValueError, match="not valid"):
        monte_carlo_step_label(task, [], "nonsense", 2, seed=1)


def test_build_step_examples_has_real_soft_and_hard_labels(dataset):
    examples = build_step_examples(dataset.math_train[:1], rollouts=16, seed=3)
    assert len(examples) == 30
    assert any(0 < example.label < 1 for example in examples)
    assert any(example.label == 0 for example in examples)
    assert any(example.label == 1 for example in examples)

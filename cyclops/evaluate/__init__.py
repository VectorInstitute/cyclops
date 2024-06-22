"""Evaluate package."""

from cyclops.evaluate.evaluator import evaluate
from cyclops.evaluate.fairness.evaluator import evaluate_fairness


__all__ = ["evaluate", "evaluate_fairness"]

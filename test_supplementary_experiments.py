"""Fast unit tests for the public supplementary-experiment utilities."""

from __future__ import annotations

import unittest

import torch

from supplementary_experiments import (
    RAW_FEATURE_NAMES,
    StableFocalLoss,
    canonical_parameter_value,
    make_feature_layout,
    select_columns,
    stage2_ablation_definitions,
)


class SupplementaryExperimentTests(unittest.TestCase):
    def test_fractional_focal_loss_has_finite_gradient(self) -> None:
        logits = torch.tensor(
            [[1000.0, -1000.0], [-1000.0, 1000.0], [0.0, 0.0]],
            requires_grad=True,
        )
        labels = torch.tensor([0, 1, 1])
        loss = StableFocalLoss(torch.ones(2), gamma=0.5)(logits, labels)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_feature_layout_is_complete_and_non_overlapping(self) -> None:
        layout = make_feature_layout(input_dim=10, graph_dim=512, layer_dim=32)
        columns = [column for group in layout.groups.values() for column in group]
        self.assertEqual(len(columns), 558)
        self.assertEqual(len(set(columns)), 558)
        self.assertEqual(set(columns), set(range(558)))

    def test_required_leave_one_group_out_ablations_exist(self) -> None:
        definitions = stage2_ablation_definitions()
        required = {
            "full",
            "full_minus_probability",
            "full_minus_raw",
            "full_minus_graph",
            "full_minus_level",
            "full_minus_neighbourhood",
        }
        self.assertTrue(required.issubset(definitions))
        self.assertEqual(len(definitions["full"]), 5)
        for name in required - {"full"}:
            self.assertEqual(len(definitions[name]), 4)

    def test_per_raw_feature_leave_one_out_selects_all_but_one(self) -> None:
        definitions = stage2_ablation_definitions()
        layout = make_feature_layout(input_dim=10, graph_dim=512, layer_dim=32)
        full_columns = set(select_columns(layout, definitions["full"]))
        self.assertEqual(len(full_columns), 558)
        for name in RAW_FEATURE_NAMES:
            key = f"full_minus_raw_{name}"
            self.assertIn(key, definitions)
            columns = set(select_columns(layout, definitions[key]))
            removed_index = RAW_FEATURE_NAMES.index(name)
            self.assertEqual(columns, full_columns - {removed_index})
            self.assertEqual(len(columns), 557)

    def test_numeric_checkpoint_values_are_canonical(self) -> None:
        self.assertEqual(canonical_parameter_value(64), canonical_parameter_value(64.0))
        self.assertEqual(canonical_parameter_value(5e-4), "0.0005")


if __name__ == "__main__":
    unittest.main()

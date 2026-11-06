import json
import unittest
from pathlib import Path

import numpy as np

from one_pixel_attack import apply_candidates, one_pixel_attack
from one_pixel_attack.differential_evolution import mutate, select
from one_pixel_attack.utils import summarize


def sensitive_predict(images):
    # Deliberately vulnerable synthetic classifier, not a CIFAR-10 benchmark.
    p = images[:, 0, 0, 0].astype(float) / 255
    return np.column_stack([1 - p, p])


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((1, 1, 3), dtype=np.uint8)

    def test_perturbation_clips_rounds_and_preserves_source(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        actual = apply_candidates(image, [[-5, 99, 300, -1, 12.6], [1, 2, 2, 3, 4]])
        np.testing.assert_array_equal(actual[0, 0, 3], [255, 0, 13])
        self.assertEqual(np.any(actual[1] != image, axis=2).sum(), 1)
        self.assertFalse(image.any())
        self.assertFalse(np.shares_memory(actual, image))

    def test_donors_distinct_and_exclude_parent(self):
        class Donors:
            def __init__(self):
                self.i = 0

            def choice(self, choices, size, replace):
                self_outer.assertNotIn(self.i, choices)
                self_outer.assertEqual(size, 3)
                self_outer.assertFalse(replace)
                self.i += 1
                return choices[:3]

        self_outer = self
        p = np.arange(20, dtype=float).reshape(4, 5)
        result = mutate(p, np.full(5, 100), 0.5, Donors())
        np.testing.assert_array_equal(result[0], p[1] + 0.5 * (p[2] - p[3]))

    def test_selection_retains_parent_on_tie(self):
        p, f = np.zeros((4, 5)), np.array([1.0, 2, 3, 4])
        select(p, f, np.ones((4, 5)), np.array([0.0, 2, 4, 3]))
        np.testing.assert_array_equal(p[:, 0], [1, 0, 0, 1])
        np.testing.assert_array_equal(f, [0, 2, 3, 3])

    def test_targeted_and_untargeted(self):
        for target in [None, 1]:
            result = one_pixel_attack(
                self.image,
                0,
                sensitive_predict,
                target_label=target,
                population_size=12,
                generations=5,
                seed=42,
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["adversarial_prediction"], 1)
            self.assertEqual(result["changed_pixels"], 1)
            self.assertEqual(result["attack_queries"], 12)
            self.assertEqual(result["total_queries"], 13)

    def test_exact_query_accounting_and_budget(self):
        calls = []

        def predict(images):
            calls.append(len(images))
            return np.tile([0.9, 0.1], (len(images), 1))

        for method in ["de", "random"]:
            calls.clear()
            result = one_pixel_attack(
                self.image,
                0,
                predict,
                population_size=8,
                generations=3,
                stop_rule="budget",
                method=method,
            )
            self.assertEqual(result["attack_queries"], 32)
            self.assertEqual(sum(calls), result["total_queries"])
            self.assertEqual(result["generations"], 3)
            self.assertFalse(result["success"])
            self.assertTrue(np.all(np.diff(result["fitness_history"]) <= 0))

    def test_zero_generations_still_evaluates_initial_population(self):
        result = one_pixel_attack(
            self.image, 0, sensitive_predict, population_size=4, generations=0
        )
        self.assertEqual(result["attack_queries"], 4)
        self.assertEqual(result["generations"], 0)

    def test_deterministic(self):
        a = one_pixel_attack(
            self.image,
            0,
            sensitive_predict,
            population_size=8,
            generations=5,
            stop_rule="budget",
        )
        b = one_pixel_attack(
            self.image,
            0,
            sensitive_predict,
            population_size=8,
            generations=5,
            stop_rule="budget",
        )
        np.testing.assert_array_equal(a["candidate"], b["candidate"])
        self.assertEqual(a["fitness_history"], b["fitness_history"])

    def test_preserve_success_even_if_unsuccessful_candidate_has_better_objective(self):
        calls = 0

        def predict(images):
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.array([[0.8, 0.1, 0.1]])
            probabilities = np.tile([0.34, 0.33, 0.33], (len(images), 1))
            probabilities[0] = [0.4, 0.5, 0.1]
            return probabilities

        result = one_pixel_attack(
            self.image, 0, predict, population_size=4, generations=0
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["adversarial_prediction"], 1)
        self.assertAlmostEqual(result["fitness_history"][0], 0.34)

    def test_invalid_inputs(self):
        cases = [
            dict(population_size=3),
            dict(generations=-1),
            dict(generations=1.5),
            dict(mutation_factor=float("nan")),
            dict(target_label=0),
            dict(target_label=2),
            dict(stop_rule="unknown"),
            dict(method="unknown"),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                one_pixel_attack(self.image, 0, sensitive_predict, **kwargs)
        with self.assertRaises(ValueError):
            one_pixel_attack(self.image.astype(float), 0, sensitive_predict)
        with self.assertRaises(ValueError):
            one_pixel_attack(self.image, 1, sensitive_predict)
        with self.assertRaises(ValueError):
            one_pixel_attack(self.image, 0, lambda x: np.full((len(x), 2), np.nan))

    def test_invalid_candidate_probabilities(self):
        def predict(images):
            return np.tile(
                [0.9, 0.1] if len(images) == 1 else [0.9, 0.9], (len(images), 1)
            )

        with self.assertRaises(ValueError):
            one_pixel_attack(self.image, 0, predict, population_size=4)

    def test_notebook_parity(self):
        notebook = json.loads(
            (Path(__file__).parents[1] / "notebooks/one-pixel-attack.ipynb").read_text(
                encoding="utf-8"
            )
        )
        namespace = {"np": np}
        # Execute only the two known, retained algorithm cells, without downloads.
        exec("".join(notebook["cells"][13]["source"]), namespace)
        exec("".join(notebook["cells"][15]["source"]), namespace)
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        for target in [None, 1]:
            for stop in ["budget", "success", "paper"]:
                params = dict(
                    image=image,
                    true_label=0,
                    predict=sensitive_predict,
                    target_label=target,
                    population_size=12,
                    generations=6,
                    seed=123,
                    stop_rule=stop,
                )
                a = namespace["one_pixel_attack"](**params)
                b = one_pixel_attack(**params)
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    np.testing.assert_equal(a[key], b[key], err_msg=key)

    def test_metrics_denominators(self):
        records = [
            dict(
                label=0,
                before=0,
                after=1,
                attacked=True,
                success=True,
                attack_queries=8,
                total_queries=9,
                pixel_mae=20,
            ),
            dict(label=0, before=1, after=1, attacked=False, success=False),
        ]
        result = summarize(records)
        self.assertEqual(result["attack_success_rate"], 1)
        self.assertEqual(result["accuracy_before"], 0.5)
        self.assertEqual(result["accuracy_after"], 0)
        self.assertEqual(result["evaluation_queries"], 11)
        empty = summarize(records[1:])
        self.assertIsNone(empty["attack_success_rate"])

    def test_saved_notebook_metrics_arithmetic(self):
        data = json.loads(
            (Path(__file__).parents[1] / "results/notebook/provenance.json").read_text()
        )
        for model in ["resnet18", "vgg16"]:
            row = data[model]
            self.assertAlmostEqual(
                row["attempted"] / row["sample_size"], row["accuracy_before"]
            )
            self.assertAlmostEqual(
                (row["attempted"] - row["successful"]) / row["sample_size"],
                row["accuracy_after"],
            )


if __name__ == "__main__":
    unittest.main()

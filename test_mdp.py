import unittest
from mdp import MDP
import numpy as np


class TestMDP(unittest.TestCase):
    def test_build_mdp(self):
        m = MDP("/Users/robpitkin/Desktop/mdp-dynamic-programming/mdpTest.txt")
        states = ["s0", "s1", "s2", "s3"]
        actions = ["a01", "a02", "a10", "a13", "a20", "a23", "a31", "a32", "a33"]
        probabilities = {
            "a01": np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a02": np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a10": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a13": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a20": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a23": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a31": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ),
            "a32": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            "a33": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        }
        rewards = {
            ("s0", "a01"): -2.0,
            ("s0", "a02"): -3.0,
            ("s1", "a10"): -1.0,
            ("s1", "a13"): -4.0,
            ("s2", "a20"): -1.0,
            ("s2", "a23"): -4.0,
            ("s3", "a31"): -2.0,
            ("s3", "a32"): -3.0,
            ("s3", "a33"): 0,
        }
        discount_factor = 0.9
        self.assertListEqual(states, m.states)
        self.assertListEqual(actions, m.actions)
        for l in probabilities:
            self.assertListEqual(probabilities[l].tolist(), m.probabilities[l].tolist())
        self.assertDictEqual(rewards, m.rewards)
        self.assertEqual(discount_factor, m.gamma)


if __name__ == "__main__":
    unittest.main()

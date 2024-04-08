import unittest
from mdp import MDP
import numpy as np


class TestMDP(unittest.TestCase):
    def test_build_mdp(self):
        m = MDP("/Users/robpitkin/Desktop/mdp-dynamic-programming/mdpTest.txt")
        states = ["s0", "s1", "s2", "s3"]
        actions = ["up", "down", "left", "right", "stay"]
        probabilities = {
            "up": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ),
            "down": np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "left": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            "right": np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "stay": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        }
        rewards = {
            ("s0", "right"): -2.0,
            ("s0", "down"): -3.0,
            ("s1", "left"): -1.0,
            ("s1", "down"): -4.0,
            ("s2", "up"): -1.0,
            ("s2", "right"): -4.0,
            ("s3", "up"): -2.0,
            ("s3", "left"): -3.0,
            ("s3", "stay"): 0,
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

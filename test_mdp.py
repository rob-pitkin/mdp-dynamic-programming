import unittest
from mdp import MDP
import numpy as np


class TestMDP(unittest.TestCase):
    def test_build_mdp(self):
        m = MDP("/Users/robpitkin/Desktop/mdp-dynamic-programming/mdpTest.txt")
        states = ["s1", "s2", "s3", "s4"]
        actions = ["a12", "a13", "a21", "a24", "a31", "a34", "a42", "a43", "a44"]
        probabilities = {
            "a12": np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a13": np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a21": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a24": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a31": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a34": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "a42": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ),
            "a43": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            "a44": np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        }
        rewards = {
            ("s1", "a12"): -2.0,
            ("s1", "a13"): -3.0,
            ("s1", "a21"): 0,
            ("s1", "a24"): 0,
            ("s1", "a31"): 0,
            ("s1", "a34"): 0,
            ("s1", "a42"): 0,
            ("s1", "a43"): 0,
            ("s1", "a44"): 0,
            ("s2", "a12"): 0,
            ("s2", "a13"): 0,
            ("s2", "a21"): -1.0,
            ("s2", "a24"): -4.0,
            ("s2", "a31"): 0,
            ("s2", "a34"): 0,
            ("s2", "a42"): 0,
            ("s2", "a43"): 0,
            ("s2", "a44"): 0,
            ("s3", "a12"): 0,
            ("s3", "a13"): 0,
            ("s3", "a21"): 0,
            ("s3", "a24"): 0,
            ("s3", "a31"): -1.0,
            ("s3", "a34"): -4.0,
            ("s3", "a42"): 0,
            ("s3", "a43"): 0,
            ("s3", "a44"): 0,
            ("s4", "a12"): 0,
            ("s4", "a13"): 0,
            ("s4", "a21"): 0,
            ("s4", "a24"): 0,
            ("s4", "a31"): 0,
            ("s4", "a34"): 0,
            ("s4", "a42"): -2.0,
            ("s4", "a43"): -3.0,
            ("s4", "a44"): 0,
        }
        discount_factor = 0.9
        self.assertListEqual(states, m.states)
        self.assertListEqual(actions, m.actions)
        for l in probabilities:
            self.assertListEqual(probabilities[l].tolist(), m.probabilities[l].tolist())
        self.assertDictEqual(rewards, m.rewards)
        self.assertEqual(discount_factor, m.gamma)

if __name__ == '__main__':
    unittest.main()


# MDP Class in Python using dictionaries
# Author: Rob Pitkin
# Date: 03/28/24

from typing import Dict
import numpy as np


class MDP:
    def __init__(self, filepath: str):
        self.mdp = self.build_mdp(filepath)
        self.states = self.mdp['s']
        self.actions = self.mdp['a']
        self.probabilities = self.mdp['p']
        self.rewards = self.mdp['r']
        self.gamma = self.mdp['y']

    def build_mdp(self, filepath: str) -> Dict[str, object]:
        """
        Build MDP expects a filepath containing a serialized MDP
        of the format:
        s1,s2,s3....                   <- states
        a1,a2,a3....                   <- actions
        s1 a1 s2 0.5, s1 a2 s2 0.5,... <- probabilities
        s1 a1 1.0, s2 a2 2.0,...       <- rewards
        0.9                            <- gamma

        Parameters:
            filepath (str): filepath of the serialized MDP

        Returns:
            Dict[str, object]: a dictionary of <S,A,P,R,y>
        """
        mdp = dict()
        try:
            lines = []
            with open(filepath, "r") as file:
                for line in file:
                    lines.append(line.strip())

            mdp["s"] = lines[0].split(",")
            assert len(mdp["s"]) > 0, "Num of states must be > 0"
            print('Parsed states')

            mdp["a"] = lines[1].split(",")
            assert len(mdp["a"]) > 0, "Num of actions must be > 0"
            print('Parsed actions')

            mdp["p"] = {a: np.zeros(shape=(len(mdp["s"]), len(mdp["s"]))) for a in mdp["a"]}
            probs_strings = lines[2].split(",")
            for s in probs_strings:
                sasp = s.split(" ")
                assert (
                    len(sasp) == 4
                ), "Probability transitions must be formatted as 's1 a1 s2 p'"
                mdp["p"][sasp[1]][int(sasp[0][-1]) - 1][int(sasp[2][-1]) - 1] = float(sasp[3])
            assert len(mdp["p"]) > 0, "Num of probability transitions must be > 0"
            assert (
                len(mdp["p"]) == len(mdp["a"])
            ), "Num of probability transition matrices must equal |a|"
            for a in mdp["a"]:
                assert mdp["p"][a].shape == (len(mdp["s"]), len(mdp["s"]))
            print('Parsed probabilities')

            mdp["r"] = dict()
            rewards_strings = lines[3].split(",")
            for s in rewards_strings:
                sar = s.split(" ")
                assert len(sar) == 3, "Rewards must be formatted as 's1 a1 r'"
                mdp["r"][(sar[0], sar[1])] = float(sar[2])
            assert len(mdp["r"]) > 0, "Num of rewards must be > 0"
            assert len(mdp["r"]) == len(mdp["s"]) * len(
                mdp["a"]
            ), "Num of rewards must equal s * a"
            print('Parsed rewards')

            mdp["y"] = float(lines[4])
            assert (
                mdp["y"] >= 0 and mdp["y"] <= 1.0
            ), "Discount factor must be within [0-1]"
            print("Parsed discount factor")

            return mdp
        except FileNotFoundError:
            print(f"Error: File with path {filepath} not found")
        except Exception as e:
            print(f"An error occurred while processing the MDP file: {e}")

        return None

    def print_mdp(self):
        print("States:",self.states)
        print("Actions:",self.actions)
        print("Probabilities:",self.probabilities)
        print("Rewards:",self.rewards)
        print("Gamma:",self.gamma)

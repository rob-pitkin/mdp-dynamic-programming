# Implementing Policy Evaluation
# Author: Rob Pitkin
# Date: 03/28/24

import numpy as np
from typing import Dict
from mdp import MDP
import pandas as pd


def policy_evaluation(
    policy: Dict[str, any], state_values: any, mdp: Dict[str, any], epsilon: float = 0.0
):
    """
    Performs a single iteration of policy evaluation for a given policy.

    Parameters:
        policy (Dict[str, (str, float)]): A policy mapping states to action, action prob pairs
        state_values (np.array): A state value vector mapping states to values
        mdp (Dict[str, any]): An MDP represented by a dictionary of states, actions, rewards, etc. mapping to their respective values
        episilon (float): The threshold to compare v(s) with v'(s) against to see if we are done evaluating

    Returns:
        state_values_new (Dict[str, float]): The new state value function
        did_change (bool): Whether or the value function changed > epsilon
    """
    try:
        state_values_new = np.copy(state_values)
        did_change = False
        for i in range(len(state_values)):
            state = f"s{i}"
            value = 0
            for action, action_probability in policy[state]:
                next_state_values = np.sum(mdp["p"][action][i] * state_values)
                value += action_probability * (
                    mdp["r"][(state, action)] + mdp["y"] * next_state_values
                )
            state_values_new[i] = value
        if np.max(np.abs(state_values_new - state_values)) > epsilon:
            did_change = True
        return state_values_new, did_change
    except Exception as e:
        print(f"An error occurred during policy evaluation: {e}")


def main():
    mdp = MDP("/Users/robpitkin/Desktop/mdp-dynamic-programming/mdp2.txt")
    state_values = np.zeros(len(mdp.mdp["s"]))
    policy = {s: [] for s in mdp.states}
    for s in mdp.states:
        prob = 1.0 / len(mdp.reachable_states[s])
        for s2 in mdp.reachable_states[s]:
            s_num = int(s[1:])
            s2_num = int(s2[1:])
            if s_num == s2_num - 4:
                policy[s].append(("down", prob))
            elif s_num == s2_num + 4:
                policy[s].append(("up", prob))
            elif s_num == s2_num - 1:
                policy[s].append(("right", prob))
            elif s_num == s2_num + 1:
                policy[s].append(("left", prob))
            elif s_num == s2_num:
                policy[s].append(("stay", prob))
            else:
                raise KeyError
    did_change = True
    before, after = None, None
    iter_num = 0
    while did_change:
        iter_num += 1
        before = state_values
        state_values, did_change = policy_evaluation(policy, state_values, mdp.mdp, 0.1)
        after = state_values
    print(f"POLICY EVAL TOOK {iter_num} ITERATIONS TO CONVERGE WITH EPSILON {0.01}")
    print("BEFORE:\n", pd.DataFrame(np.resize(before, (4,4))))
    print("AFTER:\n", pd.DataFrame(np.resize(after, (4,4))))
    return


if __name__ == "__main__":
    main()

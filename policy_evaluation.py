# Implementing Policy Evaluation
# Author: Rob Pitkin
# Date: 03/28/24

import numpy as np
from typing import Dict
from mdp import MDP


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
        if abs(np.max(state_values_new[i] - state_values[i])) > epsilon:
            did_change = True
        return state_values_new, did_change
    except Exception as e:
        print(f"An error occurred during policy evaluation: {e}")


def main():
    mdp = MDP("/Users/robpitkin/Desktop/mdp-dynamic-programming/mdp1.txt")
    state_values = np.zeros(len(mdp.mdp["s"]))
    policy = {s: [] for s in mdp.states}
    for s in mdp.states:
        prob = 1.0 / len(mdp.reachable_states[s])
        for s2 in mdp.reachable_states[s]:
            policy[s].append((f"a{s[1:]}{s2[1:]}", prob))
    print("BEFORE:", state_values)
    state_values, did_change = policy_evaluation(policy, state_values, mdp.mdp, 0.01)
    print("AFTER:", state_values)
    print("BEFORE:", state_values)
    state_values, did_change = policy_evaluation(policy, state_values, mdp.mdp, 0.01)
    print("AFTER:", state_values)
    return


if __name__ == "__main__":
    main()

# Implementing Policy Evaluation
# Author: Rob Pitkin
# Date: 04/03/24

import copy
import numpy as np
from typing import Dict
from mdp import MDP
import pandas as pd
from policy_evaluation import policy_evaluation

def policy_iteration(policy: Dict[str, any], state_values: any, mdp: Dict[str, any]) -> Dict[str, any]:
    """
    Performs a single iteration of policy iteration for a given policy.

    Parameters:
        policy (Dict[str, (str, float)]): A policy mapping states to action, action prob pairs
        state_values (np.array): A state value vector mapping states to values
        mdp (Dict[str, any]): An MDP represented by a dictionary of states, actions, rewards, etc. mapping to their respective values

    Returns:
        new_policy (Dict[str, (str, float)]): The new policy found by argmaxing over all actions for each state
    """
    new_policy = {s: a for s, a in policy.items()}
    for s in policy.keys():
        max_a = None
        max_i = 0
        max_val = -np.inf
        for i, (a, _) in enumerate(policy[s]):
            r_sum = 0.0
            probs = mdp['p'][a]
            reward = mdp['r'][(s,a)]
            gamma = mdp['y']
            state = int(s[1:])
            for state_prime, _ in enumerate(probs[state]):
                r_sum += probs[state][state_prime]*(reward + gamma*state_values[state_prime])
            if r_sum > max_val:
                max_val = r_sum
                max_a = a
                max_i = i
            new_policy[s][i] = (a, 0.0)
        new_policy[s][max_i] = (max_a, 1.0)
    return new_policy

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
    before, after = None, None
    did_change = True
    iter_num = 0
    while did_change:
        iter_num += 1
        before = state_values
        state_values, did_change = policy_evaluation(policy, state_values, mdp.mdp, 0.1)
        after = state_values
    print(f"POLICY EVAL TOOK {iter_num} ITERATIONS TO CONVERGE WITH EPSILON {0.01}")
    print("BEFORE:\n", pd.DataFrame(np.resize(before, (4,4))))
    print("AFTER:\n", pd.DataFrame(np.resize(after, (4,4))))
    print('POLICY BEFORE:', policy)
    policy = policy_iteration(policy, state_values, mdp.mdp)
    print('POLICY AFTER:', policy)
    before, after = None, None
    did_change = True
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

if __name__ == '__main__':
    main()
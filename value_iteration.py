# Implementing Value Evaluation
# Author: Rob Pitkin
# Date: 04/03/24

import numpy as np
from typing import Dict
from mdp import MDP
import pandas as pd

def value_iteration(state_values: any, mdp: MDP, epsilon: float) -> any:
    """
    Performs a single iteration of value iteration for a given state-value function.

    Parameters:
        state_values (np.array): A state value vector mapping states to values
        mdp (Dict[str, any]): An MDP represented by a dictionary of states, actions, rewards, etc. mapping to their respective values
        epsilon (float): The threshold to compare v(s) with v'(s) against to see if we are done evaluating

    Returns:
        new_values (np.array): The new state value function
        did_change (bool): Whether or not the state value fn changed
    """
    # v(s) = max_a(R_{a,s} + gamma * sum_s(P_{a,s,s'} * v(s')))
    new_values = np.zeros_like(state_values)
    did_change = False
    rewards = mdp['r']
    probs = mdp['p']
    gamma = mdp['y']
    actions = mdp['a']
    for i in range(len(state_values)):
        max_val = -np.inf
        for a in actions:
            if (f's{i}', a) in rewards.keys():
                max_val = max(rewards[(f's{i}', a)] + gamma * probs[a][i] @ state_values, max_val)
        new_values[i] = max_val
    if max(abs(state_values - new_values)) > epsilon:
        did_change = True
    return new_values, did_change

def main():
    mdp = MDP("/Users/robpitkin/Desktop/mdp-dynamic-programming/mdp2.txt")
    state_values = np.zeros(len(mdp.mdp["s"]))
    before, after = None, None
    did_change = True
    iter_num = 0
    epsilon = 0.01
    while did_change:
        iter_num += 1
        before = state_values
        state_values, did_change = value_iteration(state_values, mdp.mdp, epsilon)
        after = state_values
    print(f"VALUE ITER TOOK {iter_num} ITERATIONS TO CONVERGE WITH EPSILON {epsilon}")
    print("BEFORE:\n", pd.DataFrame(np.resize(before, (4,4))))
    print("AFTER:\n", pd.DataFrame(np.resize(after, (4,4))))
    return
        

if __name__ == '__main__':
    main()
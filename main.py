"""
This Python script implements a Q-learning algorithm for training an agent to solve the Taxi-v3
environment from the OpenAI Gym library. It includes functions for sampling actions based on
epsilon-greedy policy, calculating Q-values, performing training steps, and solving the environment
with a trained Q-table.
"""

import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def sample_action(env, epsilon, obs_q_values):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()

    return np.argmax(obs_q_values)

def action_from_obs(obs, q_table, q_params, env):
    obs_q_values = q_table[obs]
    epsilon = q_params["epsilon"]

    return sample_action(env, epsilon, obs_q_values)

def calc_q_value(old_q_value, reward, max_next_q_value, alpha, gamma):
    return (1 - alpha) * old_q_value + alpha * (reward + gamma * max_next_q_value)

def q_value_from_action(action, q_table, obs, new_obs, reward, q_params):
    old_q_value = q_table[obs][action]
    new_obs_q_values = q_table[new_obs]
    max_next_q_value = np.max(new_obs_q_values)
    alpha = q_params["alpha"]
    gamma = q_params["gamma"]

    return calc_q_value(old_q_value, reward, max_next_q_value, alpha, gamma)

def do_step(q_table, obs, env, q_params):
    action = action_from_obs(obs, q_table, q_params, env)
    new_obs, reward, terminated, _, _ = env.step(action)
    q_table[obs, action] = q_value_from_action(action, q_table, obs, new_obs, reward, q_params)

    return terminated, new_obs, q_table

def train_q_table(env, q_params, num_iter):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    for _ in tqdm(range(num_iter), desc="Training Q-table..."):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            terminated, obs, q_table = do_step(q_table, obs, env, q_params)

    return q_table

def solve_obs(obs, q_table, env, parameters):
    terminated = False
    solution_steps = [env.render()]
    while not terminated:
        terminated, obs, q_table = do_step(q_table, obs, env, parameters)
        solution_step = env.render()
        solution_steps.append(solution_step)

    return solution_steps

def print_solution(solution_steps):
    for step_num, step in enumerate(solution_steps):
        print(f"Step {step_num + 1}:")
        print(step)

def init():
    env = gym.make("Taxi-v3", render_mode="ansi")
    q_params = {
        "alpha": 0.1,
        "gamma": 0.6,
        "epsilon": 0.1
    }

    num_iter = 100_000
    q_table = train_q_table(env, q_params, num_iter)

    initial_obs, _ = env.reset()
    solution_steps = solve_obs(initial_obs, q_table, env, q_params)
    print_solution(solution_steps)

if __name__ == "__main__":
    init()

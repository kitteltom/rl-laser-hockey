# Code based on https://github.com/sfujim/TD3, with major modifications.

import numpy as np
import torch
import argparse
import os

from TD3 import TD3
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import laserhockey.hockey_env as h_env


def eval_policy(policy, seed, max_episode_timesteps, eval_episodes=20):
    # Set up the evaluation environment
    eval_env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    eval_env.seed(seed + 100)
    player2 = h_env.BasicOpponent(weak=False)

    # Evaluate
    eval_rewards = []
    eval_results = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state2 = eval_env.obs_agent_two()
        episode_reward = 0
        episode_result = 0

        for _ in range(max_episode_timesteps):
            action = policy.act(np.array(state))
            action2 = player2.act(np.array(state2))

            state, reward, done, info = eval_env.step(np.hstack([action, action2]))
            state2 = eval_env.obs_agent_two()
            episode_reward += reward + info["reward_touch_puck"] + info["reward_puck_direction"]
            episode_result += reward - info["reward_closeness_to_puck"]

            if done:
                break

        eval_rewards.append(episode_reward)
        eval_results.append(episode_result)

    # Print mean and std deviation of episode rewards
    print("--------------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {np.mean(eval_rewards):.3f} +- {np.std(eval_rewards):.3f}")
    print("--------------------------------------------")

    return eval_rewards, eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help='Policy name (TD3)')
    parser.add_argument("--env", default="Hockey-v0_NORMAL", help='Gym environment name')
    parser.add_argument("--trial", default=0, type=int, help='Trial number')
    parser.add_argument("--seed", default=42, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--start_timesteps", default=5e4, type=int, help='Time steps initial random policy is used')
    parser.add_argument("--eval_freq", default=5e3, type=int, help='How often (time steps) it will be evaluated')
    parser.add_argument("--self_play_freq", default=15e5, type=int, help='Add current agent to list of opponents')
    parser.add_argument("--max_timesteps", default=1e6, type=int, help='Max time steps to run environment')
    parser.add_argument("--max_episode_timesteps", default=500, type=int, help='Max time steps per episode')
    parser.add_argument("--max_buffer_size", default=1e6, type=int, help='Size of the replay buffer')
    parser.add_argument("--expl_noise", default=0.15, type=float, help='Std of Gaussian exploration noise')
    parser.add_argument("--hidden_dim", default=256, type=int, help='Hidden dim of actor and critic nets')
    parser.add_argument("--batch_size", default=256, type=int, help='Batch size for both actor and critic')
    parser.add_argument("--learning_rate", default=3e-4, type=float, help='Learning rate')
    parser.add_argument("--discount", default=0.99, type=float, help='Discount factor')
    parser.add_argument("--tau", default=0.01, type=float, help='Target network update rate')
    parser.add_argument("--policy_noise", default=0.1, type=float,
                        help='Noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.5, type=float, help='Range to clip target policy noise')
    parser.add_argument("--policy_freq", default=2, type=int, help='Frequency of delayed policy updates')
    parser.add_argument("--prioritized_replay", action="store_true", help='Use prioritized experience replay')
    parser.add_argument("--alpha", default=0.6, type=float, help='Amount of prioritization in PER')
    parser.add_argument("--beta", default=1.0, type=float, help='Amount of importance sampling in PER')
    parser.add_argument("--beta_schedule", default="", help='Annealing schedule for beta in PER')
    parser.add_argument("--normalize_obs", action="store_true", help='Use observation normalisation')
    parser.add_argument("--only_win_reward", action="store_true", help='Rewards only wins')
    parser.add_argument("--early_stopping", action="store_true", help='Use early stopping')
    parser.add_argument("--load_model", default="",
                        help='Model load file name, \"\" does not load')
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.trial}"
    print("--------------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("--------------------------------------------")

    # Create folders for evaluation
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    # Create the Environment
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2  # The policy only controls the left player
    max_action = float(env.action_space.high[0])

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "max_action": max_action,
        "lr": args.learning_rate,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "normalize_obs": args.normalize_obs
    }
    policy = TD3(**kwargs)

    # Create other players
    opponent_policies = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)]
    warm_up_player = h_env.BasicOpponent(weak=False)

    # Load previous model if applicable
    if args.load_model != "":
        policy.load(f"./models/{args.load_model}")
        warm_up_player = policy

        # Also add self to list of opponents
        old_self = TD3(**kwargs)
        old_self.load(f"./models/{args.load_model}")
        opponent_policies.append(old_self)

    # Replay buffer
    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(
            state_dim,
            action_dim,
            max_size=int(args.max_buffer_size),
            total_t=int(args.max_timesteps),
            alpha=args.alpha,
            beta=args.beta,
            beta_schedule=args.beta_schedule
        )
    else:
        replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(args.max_buffer_size))

    # Create evaluation data structure
    evaluations = {
        'train_rewards': [],
        'eval_rewards': [],
        'eval_results': [],
        'final_eval_rewards': [],
        'final_eval_results': []
    }

    # Initialize the environment
    state, done = env.reset(), False
    state2 = env.obs_agent_two()
    player2 = np.random.choice(opponent_policies)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    best_policy = {
        't': 0,
        'average_return': 0
    }

    for t in range(int(args.start_timesteps + args.max_timesteps)):

        episode_timesteps += 1

        # Select action
        if t < args.start_timesteps:
            action = warm_up_player.act(np.array(state))
            action2 = player2.act(np.array(state2))
        else:
            action = (
                policy.act(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            action2 = player2.act(np.array(state2))

        # Perform action (composed of the policy's and the opponent's action) and collect reward
        next_state, reward, done, info = env.step(np.hstack([action, action2]))
        reward += info["reward_touch_puck"] + info["reward_puck_direction"]
        if args.only_win_reward:
            reward = 10.0 if info["winner"] == 1 else 0.0
        done_bool = float(done)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        # Update the state and return
        state = next_state
        state2 = env.obs_agent_two()
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.prioritized_replay, args.batch_size)

        if done or episode_timesteps >= args.max_episode_timesteps:
            # Print episode info
            print(
                f"Total T: {t + 1}, " +
                f"Episode Num: {episode_num + 1}, " +
                f"Episode T: {episode_timesteps}, " +
                f"Reward: {episode_reward:.3f}"
            )

            if t >= args.start_timesteps:
                evaluations['train_rewards'].append(episode_reward)
                episode_num += 1

            # Reset environment
            state, done = env.reset(), False
            state2 = env.obs_agent_two()
            player2 = np.random.choice(opponent_policies)
            episode_reward = 0
            episode_timesteps = 0

        # Evaluate the policy
        if (t + 1) % args.eval_freq == 0 and (t + 1) >= args.start_timesteps:
            eval_rewards, eval_results = eval_policy(
                policy,
                args.seed,
                args.max_episode_timesteps
            )
            evaluations['eval_rewards'].append(eval_rewards)
            evaluations['eval_results'].append(eval_results)
            np.save(f"./results/{file_name}", evaluations)

            # Early stopping
            if not args.early_stopping:
                policy.save(f"./models/{file_name}")
            elif np.mean(eval_rewards) > best_policy['average_return']:
                best_policy['t'] = t
                best_policy['average_return'] = np.mean(eval_rewards)
                policy.save(f"./models/{file_name}")

        # Add opponent
        if (t + 1) % args.self_play_freq == 0:
            opponent_policy = TD3(**kwargs)
            opponent_policy.load(f"./models/{file_name}")
            opponent_policies.append(opponent_policy)

    # Final evaluation
    if args.early_stopping:
        print("-----------------------------------------------------")
        print(f"Average return of best policy: {best_policy['average_return']:.3f} (at t = {best_policy['t']})")
        print("-----------------------------------------------------")
    final_policy = TD3(**kwargs)
    final_policy.load(f"./models/{file_name}")
    eval_rewards, eval_results = eval_policy(
        final_policy,
        args.seed,
        args.max_episode_timesteps,
        eval_episodes=100
    )
    evaluations['final_eval_rewards'] = eval_rewards
    evaluations['final_eval_results'] = eval_results
    np.save(f"./results/{file_name}", evaluations)

    # Report moments for observation normalization
    # np.save("./results/observation_moments.npy", replay_buffer.observation_moments())


if __name__ == "__main__":
    main()

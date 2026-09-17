import argparse
import torch
import numpy as np
from pathlib import Path
from robot_nav.SIM_ENV.sim import SIM
from robot_nav.utils import get_buffer


def create_model(algo, state_dim, action_dim, max_action, device, save_every, resume=False, reward_mode="default"):
    """Factory: instantiate the requested RL algorithm with sensible defaults."""
    algo = algo.upper()
    model_name = algo if reward_mode == "default" else f"{algo}_{reward_mode}"
    common = dict(
        state_dim=state_dim, action_dim=action_dim, max_action=max_action,
        device=device, save_every=save_every, load_model=resume, model_name=model_name,
    )

    if algo == "CNNTD3":
        from robot_nav.models.CNNTD3.CNNTD3 import CNNTD3
        return CNNTD3(**common)

    elif algo == "TD3":
        from robot_nav.models.TD3.TD3 import TD3
        return TD3(**common,
                   save_directory=Path("robot_nav/models/TD3/checkpoint"),
                   load_directory=Path("robot_nav/models/TD3/checkpoint"))

    elif algo == "DDPG":
        from robot_nav.models.DDPG.DDPG import DDPG
        return DDPG(**common,
                    save_directory=Path("robot_nav/models/DDPG/checkpoint"),
                    load_directory=Path("robot_nav/models/DDPG/checkpoint"))

    elif algo == "SAC":
        from robot_nav.models.SAC.SAC import SAC
        # SAC has device before max_action — handle separately
        return SAC(state_dim=state_dim, action_dim=action_dim,
                   device=device, max_action=max_action,
                   save_every=save_every, load_model=resume, model_name=model_name,
                   save_directory=Path("robot_nav/models/SAC/checkpoint"),
                   load_directory=Path("robot_nav/models/SAC/checkpoint"))

    elif algo == "PPO":
        from robot_nav.models.PPO.PPO import PPO
        return PPO(**common,
                   save_directory=Path("robot_nav/models/PPO/checkpoint"),
                   load_directory=Path("robot_nav/models/PPO/checkpoint"))

    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from: CNNTD3, TD3, DDPG, SAC, PPO")


def main(args=None):
    """Main training function"""
    # ── CLI ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="RL Navigation Training")
    parser.add_argument("--algo", type=str, default="CNNTD3",
                        choices=["CNNTD3", "TD3", "DDPG", "SAC", "PPO"],
                        help="RL algorithm to train (default: CNNTD3)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Number of training epochs (default: 60)")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N training iterations (default: 5)")
    parser.add_argument("--no-plot", action="store_true", default=True,
                        help="Disable simulation rendering (default: True for headless)")
    parser.add_argument("--reward", type=str, default="default",
                        choices=["default", "dense", "sparse", "harsh"],
                        help="Reward function mode (default: default)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from saved checkpoint")
    cli = parser.parse_args(args)

    # ── Hyperparameters ──────────────────────────────────
    action_dim = 2
    max_action = 1
    state_dim = 185
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nr_eval_episodes = 10
    max_epochs = cli.epochs
    epoch = 0
    episodes_per_epoch = 70
    episode = 0
    train_every_n = 2
    training_iterations = 80
    batch_size = 64
    max_steps = 300
    steps = 0
    load_saved_buffer = False
    pretrain = False
    pretraining_iterations = 10

    # ── Model ────────────────────────────────────────────
    model = create_model(
        algo=cli.algo, state_dim=state_dim, action_dim=action_dim,
        max_action=max_action, device=device, save_every=cli.save_every,
        resume=cli.resume, reward_mode=cli.reward,
    )
    # Tag model/tensorboard with reward mode for separate tracking
    if cli.reward != "default":
        model.model_name = f"{cli.algo}_{cli.reward}"
        model.writer = type(model.writer)(comment=model.model_name)

    # PPO is on-policy: needs larger rollout batches, fewer iterations per batch
    if cli.algo == "PPO":
        train_every_n = 20
        training_iterations = 10
        print(f"PPO mode: train_every_n={train_every_n}, training_iterations={training_iterations}")

    print(f"Training with {cli.algo} on {device} for {max_epochs} epochs, reward={cli.reward}")

    # ── Environment & Buffer ─────────────────────────────
    sim = SIM(
        world_file="worlds/robot_world.yaml",
        disable_plotting=cli.no_plot,
        reward_mode=cli.reward,
    )
    replay_buffer = get_buffer(
        model, sim, load_saved_buffer, pretrain,
        pretraining_iterations, training_iterations, batch_size,
    )

    # ── Training Loop ────────────────────────────────────
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
        lin_velocity=0.0, ang_velocity=0.0,
    )

    while epoch < max_epochs:
        state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a,
        )
        action = model.get_action(np.array(state), True)
        a_in = [(action[0] + 1) / 4, action[1]]

        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1],
        )
        next_state, terminal = model.prepare_state(
            latest_scan, distance, cos, sin, collision, goal, a,
        )
        replay_buffer.add(state, action, reward, terminal, next_state)

        if terminal or steps == max_steps:
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
            episode += 1
            if episode % train_every_n == 0:
                model.train(
                    replay_buffer=replay_buffer,
                    iterations=training_iterations,
                    batch_size=batch_size,
                )
            steps = 0
        else:
            steps += 1

        if (episode + 1) % episodes_per_epoch == 0:
            episode = 0
            epoch += 1
            evaluate(model, epoch, sim, eval_episodes=nr_eval_episodes)


def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    for _ in range(eval_episodes):
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
        done = False
        while not done and count < 501:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a,
            )
            action = model.get_action(np.array(state), False)
            a_in = [(action[0] + 1) / 4, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1],
            )
            avg_reward += reward
            count += 1
            if collision:
                col += 1
            if goal:
                goals += 1
            done = collision or goal
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()

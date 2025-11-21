import pickle, neat, numpy as np, gymnasium as gym
from pathlib import Path
from base_training import TinyGray

import argparse


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Path to the pkl file containing champion"
        )
    )
    parser.add_argument("pkl", type=Path, help="Path to the pkl file containing champion")
    args = parser.parse_args()

    while True:
        winner_file = args.pkl
        print(winner_file.name)
        genome, config = pickle.load(winner_file.open("rb"))

        # recreates the network
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        env = TinyGray(gym.make("CarRacing-v3", render_mode="human"))
        obs, _ = env.reset(seed=9)

        done = False
        steps = 0
        while not done:
            a = np.array(net.activate(obs), dtype=np.float32)
            a[0] = np.clip(a[0], -1.0, 1.0)
            a[1:] = np.clip((a[1:] + 1.0) / 2.0, 0.0, 1.0)
            obs, _, done, _, _ = env.step(a)
            steps += 1
            if steps % 100 == 0:
                print("Steps:", steps)

        env.close()


if __name__ == "__main__":
    main()

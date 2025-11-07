#!/usr/bin/env python3

from base_training import TinyGray, SavePerGenerationReporter
import gymnasium as gym

import numpy as np
import os
import pickle

import neat
from neat import Checkpointer, StatisticsReporter, StdOutReporter
from neat.parallel import ParallelEvaluator

import multiprocessing as mp

# Configuration
MAX_STEPS = 50                   # CarRacing terminates early if you go off-track
NEAT_CONFIG_PATH = "car_neat.cfg"  # Config file for the neat-python implementation
WORKERS = None                     # None = use all CPU cores


def eval_genome(genome, config):
	"""
	Evaluation for genomes.
	Works with processing, by starting environments for each processor.
	"""
	env = TinyGray(gym.make(
		id="CarRacing-v3",
		render_mode=None,
		max_episode_steps=MAX_STEPS,
		lap_complete_percent=0.95,
		continuous=True,
		domain_randomize=False
	))

	net = neat.nn.FeedForwardNetwork.create(genome, config)
	obs, _ = env.reset(seed=None)
	total_reward  = 0.0

	for _ in range(MAX_STEPS):
		steer_raw, gas_raw, brake_raw = np.array(net.activate(obs), dtype=np.float32)

		steer: np.float32 = np.clip(steer_raw, -1.0, 1.0)

		gas: np.float32 = np.clip((gas_raw + 1.0) / 2.0, 0.0, 1.0)

		brake: np.float32 = np.clip((brake_raw + 1.0) / 2.0, 0.0, 1.0)

		action = np.array([steer, gas, brake], dtype=np.float32)

		obs, reward, terminated, truncated, _ = env.step(action)

		total_reward += reward

		if terminated or truncated:
			break

	env.close()
	return total_reward


def train_or_resume(config_path: str, generations: int, checkpoint: str | None = None):
	"""
	Either trains or resumes from a checkpoint using ParallelEvaluator
	from the neat-python implementation.
	"""

	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
						 neat.DefaultSpeciesSet, neat.DefaultStagnation,
						 config_path)

	pop = (neat.Checkpointer.restore_checkpoint(checkpoint)
		   if checkpoint else neat.Population(config))
	pop.config = config          # make sure fresh config is attached

	pop.add_reporter(StdOutReporter(True))
	stats = StatisticsReporter(); pop.add_reporter(stats)
	pop.add_reporter(SavePerGenerationReporter(MAX_STEPS=MAX_STEPS))
	os.makedirs("chk", exist_ok=True)
	pop.add_reporter(Checkpointer(10, filename_prefix="chk/car_neat-"))

	evaluator = ParallelEvaluator(WORKERS, eval_genome)

	try:
		winner = pop.run(evaluator.evaluate, generations)
	except KeyboardInterrupt:
		print("\nInterrupted — saving best genome so far …")
		winner = stats.best_genome()

	with open("champion.pkl", "wb") as f:
		pickle.dump((winner, config), f)
	print("\nBest genome fitness:", winner.fitness)


if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)

	# Fresh run:
	train_or_resume(NEAT_CONFIG_PATH, generations=1000)

	# Resume a training from a checkpoint file:
	# train_or_resume(NEAT_CONFIG_PATH, generations=1000, checkpoint="chk/car_neat-09")

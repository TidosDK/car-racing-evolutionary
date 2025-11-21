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

NEAT_CONFIG_PATH = "car_neat.cfg"  # Config file for the neat-python implementation
WORKERS = None                     # None = use all CPU cores


class neat_algorithm:
	INCREASE_MAX_STEP_EVERY_X_GENERATION = 1000  # How often it should increase the max step value
	INCREASE_MAX_STEP_BY_X = 0                   # How much it should increment the max step value by


	def __init__(self, shared_max_steps=None):
		self.max_steps = 1350

		if shared_max_steps is None:
			self.shared_max_steps = self.max_steps
		else:
			self.shared_max_steps = shared_max_steps

		self.generation_counter = 0
		self.gen_mod_increaser = 0
		self.last_increment = self.INCREASE_MAX_STEP_EVERY_X_GENERATION
		self.reporter: SavePerGenerationReporter = None

		self.reporter = SavePerGenerationReporter(max_steps=self.max_steps)
		self.reporter.algorithm = self


	def increase_max_steps(self):
		"""
		Increases the amount of steps the car can take.
		By changing the varibles "INCREASE_MAX_STEP_EVERY_X_GENERATION"
		and "INCREASE_MAX_STEP_BY_X", the increase_max_steps will automatically
		be called and change the max_steps value.
		"""
		print("Max steps:", self.max_steps)
		gen_num = self.reporter.get_gen()

		if gen_num % self.INCREASE_MAX_STEP_EVERY_X_GENERATION == 0 and self.last_increment >= (gen_num - self.INCREASE_MAX_STEP_EVERY_X_GENERATION):
			self.max_steps += self.INCREASE_MAX_STEP_BY_X
			self.reporter.set_max_steps(self.max_steps)
			self.shared_max_steps = self.max_steps
			self.last_increment = gen_num


	def eval_genome(self, genome, config):
		"""
		Evaluation for genomes.
		Works with processing, by starting environments for each processor.
		"""
		env = TinyGray(gym.make(
			id="CarRacing-v3",
			render_mode=None,
			max_episode_steps=self.shared_max_steps,
			lap_complete_percent=0.95,
			continuous=True,
			domain_randomize=False
		))

		net = neat.nn.FeedForwardNetwork.create(genome, config)
		obs, _ = env.reset(seed=9)
		total_reward  = 0.0

		for _ in range(self.shared_max_steps):
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


	def train_or_resume(self, config_path: str, generations: int, checkpoint: str | None = None):
		"""
		Either trains or resumes from a checkpoint using ParallelEvaluator
		from the neat-python implementation.
		"""

		config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation,
							config_path)

		pop = (neat.Checkpointer.restore_checkpoint(checkpoint)
			if checkpoint else neat.Population(config))
		pop.config = config

		pop.add_reporter(StdOutReporter(True))
		stats = StatisticsReporter(); pop.add_reporter(stats)
		pop.add_reporter(self.reporter)
		os.makedirs("chk", exist_ok=True)
		pop.add_reporter(Checkpointer(10, filename_prefix="chk/car_neat-"))

		evaluator = ParallelEvaluator(WORKERS, self.eval_genome)

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

	manager = mp.Manager()
	shared_max = manager.Value('i', 100)

	# Fresh run:
	algorithm = neat_algorithm(shared_max_steps=shared_max)
	algorithm.train_or_resume(config_path=NEAT_CONFIG_PATH, generations=1000)

	# Resume a training from a checkpoint file:
	# neat_algorithm.train_or_resume(NEAT_CONFIG_PATH, generations=1000, checkpoint="chk/car_neat-09")

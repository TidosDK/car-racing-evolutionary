#!/usr/bin/env python3

from base_training import TinyGray, SavePerGenerationReporter
import gymnasium as gym

import numpy as np
import os
import configparser
import argparse
from tqdm import tqdm

import neat
from neat import Checkpointer, StatisticsReporter, StdOutReporter
from neat.parallel import ParallelEvaluator

import multiprocessing as mp

from novelty import NoveltyArchive

NEAT_CONFIG_PATH = "car_neat.cfg"  # Config file for the neat-python implementation
WORKERS = None                     # None = use all CPU cores
GAME_SEED = 9                      # Seed for the car_racing map
DEBUG = False                      # Used for debugging

class CustomEvaluator:
	"""
	Custom evaluator based on the ParallelEvaluator from neat-python:
	https://github.com/CodeReclaimers/neat-python/blob/master/neat/parallel.py#L48

	It also uses multiprocessing for evaluating the algorithm.
	"""
	def __init__(self, num_workers, eval_function, mode="fitness", archive=None):
		self.num_workers = num_workers
		self.eval_function = eval_function
		self.pool = mp.Pool(processes=num_workers)
		self.mode = mode
		self.archive = archive

	def __del__(self):
		"""
		For cleaning up the multiprocessing pool when
		the CustomEvaluator is destroyed.
		"""
		if self.pool:
			self.pool.close()
			self.pool.join()

	def evaluate(self, genomes, config):
		jobs = []
		for _, genome in genomes:
			# apply_async does not interrupt the main loop.
			jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

		results = []
		for job in tqdm(jobs, desc=f"Evaluation -  mode: {self.mode}", leave=False, unit=" genome"):
			results.append(job.get())

		# Novelty:
		if self.mode == "novelty":
			population_behaviors = [res[1] for res in results]

			for i, (_, genome) in enumerate(genomes):
				reward, behavior = results[i]

				novelty_score = self.archive.calculate_novelty(behavior, population_behaviors)

				# It is necessary to use the novelty_score as the fitness value for
				# the genome, in order for neat-python to train based on this value.
				genome.fitness = novelty_score

				genome.real_reward = reward

				self.archive.update_archive(behavior, novelty_score)

			print(f"Archive Size (count of unique locations): {self.archive.size()}")

			# This print statement the Reporter from the neat-python prints information about the generation
			print("THIS FITNESS IN THE FOLLOWING INFORMATION IS BASED ON THE NOVELTY CALCULATION:")

		# Fitness-only:
		else:
			for i, (_, genome) in enumerate(genomes):
				reward, behavior = results[i]
				genome.fitness = reward


class neat_algorithm:
	INCREASE_MAX_STEP_EVERY_X_GENERATION = 1000  # How often it should increase the max step value
	INCREASE_MAX_STEP_BY_X = 0                   # How much it should increment the max step value by


	def __init__(self, shared_max_steps=None):
		self.max_steps = 500

		if shared_max_steps is None:
			self.shared_max_steps = self.max_steps
		else:
			self.shared_max_steps = shared_max_steps

		self.generation_counter = 0
		self.gen_mod_increaser = 0
		self.last_increment = self.INCREASE_MAX_STEP_EVERY_X_GENERATION
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
		obs, _ = env.reset(seed=GAME_SEED)
		total_reward  = 0.0

		final_x, final_y = 0.0, 0.0

		for _ in range(self.shared_max_steps):
			steer_raw, gas_raw, brake_raw = np.array(net.activate(obs), dtype=np.float32)

			steer: np.float32 = np.clip(steer_raw, -1.0, 1.0)

			gas: np.float32 = np.clip((gas_raw + 1.0) / 2.0, 0.0, 1.0)

			brake: np.float32 = np.clip((brake_raw + 1.0) / 2.0, 0.0, 1.0)

			action = np.array([steer, gas, brake], dtype=np.float32)

			obs, reward, terminated, truncated, _ = env.step(action)

			total_reward += reward

			try:
				final_x, final_y = env.unwrapped.car.hull.position
			except:
				if DEBUG:
					print("DEBUG: Could not extract position of car")
				pass

			if terminated or truncated:
				break

		env.close()

		return total_reward, [float(final_x), float(final_y)]


	def train_or_resume(self, config_path: str, generations: int, checkpoint: str | None = None, mode="fitness"):
		"""
		Either trains or resumes from a checkpoint using ParallelEvaluator
		from the neat-python implementation.
		"""

		config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation,
							config_path)

		parsed_config = configparser.ConfigParser()
		parsed_config.read("car_neat.cfg")

		pop = (neat.Checkpointer.restore_checkpoint(checkpoint)
			if checkpoint else neat.Population(config))
		pop.config = config

		print("\nSeed for NEAT AI:\t", parsed_config["NEAT"]["seed"])
		print("Seed for car_racing:\t", GAME_SEED)
		print("Mode:\t\t\t", mode.lower())

		pop.add_reporter(StdOutReporter(True))
		stats = StatisticsReporter()
		pop.add_reporter(stats)
		pop.add_reporter(self.reporter)
		os.makedirs("chk", exist_ok=True)
		pop.add_reporter(Checkpointer(10, filename_prefix="chk/car_neat-"))

		archive = None
		if mode.lower() == "novelty":
			archive = NoveltyArchive(threshold=5.0, k_neighbors=15)

			self.reporter.set_archive(archive)

		evaluator = CustomEvaluator(
			num_workers=WORKERS,
			eval_function=self.eval_genome,
			mode=mode,
			archive=archive
		)

		try:
			winner = pop.run(evaluator.evaluate, generations)
		except KeyboardInterrupt:
			print("\nInterrupted — saving best genome so far …")
			winner = stats.best_genome()

		print("\nBest genome fitness:", winner.fitness)


if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)

	manager = mp.Manager()
	shared_max = manager.Value('i', 100)

	parser = argparse.ArgumentParser(description='Run Car Racing Evolution')
	parser.add_argument('--mode', type=str, default='fitness', choices=['fitness', 'novelty'], help='Evolution mode: "fitness" for standard rewards, "novelty" for behavior search')
	args = parser.parse_args()

	# Fresh run:
	algorithm = neat_algorithm(shared_max_steps=shared_max)
	algorithm.train_or_resume(
		config_path=NEAT_CONFIG_PATH,
		generations=1000,
		mode=args.mode
	)

	# Resume a training from a checkpoint file:
	# neat_algorithm.train_or_resume(NEAT_CONFIG_PATH, generations=1000, checkpoint="chk/car_neat-09", mode=args.mode)

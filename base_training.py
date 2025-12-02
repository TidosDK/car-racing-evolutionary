import gymnasium as gym

import numpy as np
import cv2
import csv
import statistics
import pathlib
import pickle

from neat.reporting import BaseReporter


class TinyGray(gym.ObservationWrapper):
	"""
	The Observation Wrapper for converting observations into Gray scaled
	smaller images for the AI.
	"""
	def __init__(self, env, size=(32, 32)):
		super().__init__(env)
		self.size = size
		self.observation_space = gym.spaces.Box(
			0.0, 1.0, shape=(size[0] * size[1],), dtype=np.float32
		)


	def observation(self, obs):
		frame = cv2.cvtColor(obs[:-12], cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
		return (frame.astype(np.float32) / 255.0).flatten()


class SavePerGenerationReporter(BaseReporter):
	"""
	Class object for saving the following information:
	- Champion of each generation (best driver).
	- Best fitness, mean fitness, and worst fitness of each generation.
	"""
	def __init__(self, csv_path: str = "fitness_history.csv", champ_dir: str = "champions", max_steps: int = 1000):
		self.csv_path  = pathlib.Path(csv_path)
		self.champ_dir = pathlib.Path(champ_dir)
		self.champ_dir.mkdir(parents=True, exist_ok=True)
		self.max_steps = max_steps

		if not self.csv_path.exists():
			with self.csv_path.open("w", newline="") as f:
				csv.writer(f).writerow(["generation", "best_fitness", "mean_fitness", "worst_fitness", "max_steps"])
		self._gen = 0

	def start_generation(self, generation):
		"""
		Starts a new generation.
		Called by the implemented AI
		"""
		self._gen = generation

		if hasattr(self, "algorithm") and hasattr(self.algorithm, "increase_max_steps"):
			self.algorithm.increase_max_steps()


	def set_max_steps(self, max_steps):
		"""
		Method for changing the max steps value
		"""
		self.max_steps = max_steps


	def get_gen(self) -> int:
		return self._gen


	def end_generation(self, config, population, species_set):
		"""
		Log fitness stats and save the champion of this generation.
		Called by the implemented AI
		"""
		gen     = self._gen
		genomes = list(population.values())

		def get_distance_score(g):
			# The AI is Novelty:
			if hasattr(g, "real_reward"):
				return g.real_reward

			# The AI is fitness-only:
			if g.fitness is not None:
				return g.fitness
			return -float('inf')

		vals = [get_distance_score(g) for g in genomes]

		# Filter out invalid scores (in case of unevaluated genomes)
		valid_vals = [v for v in vals if v > -float('inf')]

		if not valid_vals:
			print(f"[Gen {gen}] no valid fitness; nothing saved."); return

		best, worst, mean = max(valid_vals), min(valid_vals), statistics.fmean(valid_vals)

		with self.csv_path.open("a", newline="") as f:
			csv.writer(f).writerow([gen, best, mean, worst, self.max_steps])

		best_genome = max(genomes, key=get_distance_score)

		champ_file  = self.champ_dir / f"gen-{gen:05d}.pkl"
		with champ_file.open("wb") as f:
			pickle.dump((best_genome, config), f)

		print(f"[Gen {gen}] champion saved → {champ_file}")

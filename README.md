# Car Racing Evolutionary Algorithm

## Start training

To start the training, you can simply run the `start_training.sh` script. It will setup the environment and start the training.

```sh
source ./start_training.sh
```

To modify how high the population is for each generation, look in the `car_neat.cfg` file. Here you can change the `pop_size` variable to what population you will like. The Python script uses multiprocessing, so each CPU core will run a generation.

## Changing parameters in the Python script

#### Same map every time

In order to make the algorithms use the same map, you can change the following line in `train_multiprocess.py`:

```
obs, _ = env.reset(seed=None)
```

Setting the `None` value to an actual integer value will set a specific map for the car to train on.

## Changing parameters in the NEAT-Python algorithm

Changes are made to the `car_neat.cfg` file.

#### Reproducibility

Changing the `seed` value in `[NEAT]` will set the seed for Python's `random` module.

```
[NEAT]
seed = 9
```

[Read more about reproducibility](https://neat-python.readthedocs.io/en/latest/reproducibility.html)

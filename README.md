# Car Racing Evolutionary Algorithm

## Start training

To start the training, you can simply run the `start_training.sh` script. It will setup the environment and start the training.

```sh
source ./start_training.sh
```

To modify how high the population is for each generation, look in the `car_neat.cfg` file. Here you can change the `pop_size` variable to what population you will like. The Python script uses multiprocessing, so each CPU core will run a generation.

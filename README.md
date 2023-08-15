# Solving Game of Life with Machine Learning
This repository contains the source code for the paper "Data-Centric Approach to Constrained Machine Learning: A Case Study on Conway's Game of Life"

### Setup
Each experiment is controlled by the configuration file in `./configs`.
Run an experiment via
```
python -m main -c config_file_name
```

For a detailed process of the training board design, see an [unhinged devlog](https://github.com/Ansebi/Game_Of_Life).

### Training Board
![game_of_life_board](https://github.com/sukiboo/game_of_life/assets/38059493/ab17205d-63e3-452a-be8c-2d20686185e2)

### Results
![success_1_recursive_tanh](https://github.com/sukiboo/game_of_life/assets/38059493/f16090ba-c116-4c98-8440-db5ca784844e)
![success_1_recursive_relu](https://github.com/sukiboo/game_of_life/assets/38059493/bad6f13b-e587-482c-98a7-4cfce39d9b55)
![success_2_recursive_tanh](https://github.com/sukiboo/game_of_life/assets/38059493/88cf79b2-8115-4f2b-866a-441739841b43)

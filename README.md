# Combining GAN and Deep RL for Sketching

Implementation of sketching agent using Deep Reinforcemenet Learning and Generative Adversarial Network.

This project is based on :
1. [Photo-Sketching](https://www.researchgate.net/publication/331591839_Photo-Sketching_Inferring_Contour_Drawings_From_Images)
2. [Doodle-SDQ](https://arxiv.org/abs/1810.05977)

## Generate Contour Drawing from Image
For generating contour drawing, This project uses code and pretrained model of Photo-Sketching available from [mtli/PhotoSkect](https://github.com/mtli/PhotoSketch).

The code is under ```img2sketch``` directory with some adjustments based on the need of this project.

## Drawing Agent
This project implements Doodle-SDQ as the method to train the drawing agent. Training process takes 2 step:
1. Supervised learning using Stroke Demonstration
  Rather than using real stroke demonstration, We use full random stroke generator to generate datasets.
  
2. Reinforcement Learning using DQN
  The pre-trained model is then retrained usin Deep RL to explore more experiences in order to learn the optimal policy.
  
## How To Use
### Requirements:
Two conda environments:
1. tfrl
  Environment with tensorflow library for the DQN agent (will update the requirements soon).
1. sketch
  Environment with PyTorch library for the sketch generator (will update the requirements soon).
### Run The Program
1. Download the pretraind model of Photo-Sketching from [mtli/PhotoSkect], unzip it, and place the files under `img2sketch/checkpoints/pretrained`.
2. Download our [final model](https://drive.google.com/file/d/11xq1w66VcxP1zGCU2owvHoq2nmFHofZx/view?usp=sharing) and place it under `models` with filename `final.h5`.
To run the program, simply run this command:
```bash
python run.py [directory to your image] [your image filename]
```

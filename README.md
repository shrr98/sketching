# Combining GAN and Deep RL for Sketching

Implementation of sketching agent using Deep Reinforcemenet Learning and Generative Adversarial Network.

This project is based on :
1. [Photo-Sketching](https://www.researchgate.net/publication/331591839_Photo-Sketching_Inferring_Contour_Drawings_From_Images)
2. [Doodle-SDQ](https://arxiv.org/abs/1810.05977)

## Generate Contour Drawing from Image
For generating contour drawing, This project uses code and pretrained model of Photo-Sketching available from [mtli/PhotoSkect](https://github.com/mtli/PhotoSketch).

[!] The code is not available yet.

## Drawing Agent
This project implements Doodle-SDQ as the method to train the drawing agent. Training process takes 2 step:
1. Supervised learning using Stroke Demonstration
  Rather than using real stroke demonstration, I use full random stroke generator to generate datasets.
  I got this result:
  ![Stroke Demo Result](test_out_bunga.avi)
  
2. Reinforcement Learning using DQN
  This is **TODO**

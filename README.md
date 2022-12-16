This is my (Pranav Talluri) Part II Project and Dissertation at the University of Cambridge.

## Completed So Far

- Implemented base UNET
- Built unconditional image generation for MNIST digits and MNIST fashion (black and white)
- Implemented ResUNET
- Built unconditional image generation for CIFAR-10 (colour)
- Implemeted FID score based evaluation
- Notes on unconditional DDPMs

## TODO

- Test impact of attention blocks
- Test FID score evaluation with larger sample size to generate comparable results with original papers
- Learn how to use HPC
- Implement methods for image masking for conditional image generation
- Notes on conditional DDPMs
- Implement EMA
- The original paper does not sweep over all hyperparameters (learning rate, batch size, ema decay factor)

## To Learn

- How to perform training and sampling with conditional DDPM
- Interpolation?

## Reading List

- Free form image inpainting with gated convolution - https://arxiv.org/abs/1806.03589
- Classifier free diffusion guidance - https://arxiv.org/abs/2207.12598
- Generative modelling with inverse heat dissapation - https://arxiv.org/abs/2206.13397

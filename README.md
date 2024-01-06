# instant-ngp-jax

Contains a concise [JAX](https://jax.readthedocs.io/en/latest/) implementation of [Vanilla NeRF](https://arxiv.org/abs/2003.08934) and an acceleration using the hash encoder described in the [Instant NGP Paper](https://arxiv.org/abs/2201.05989). Done as an exercise for learning purposes.

Prioritizes simplicity and readability over maximum performance. As such the following haven't been implemented yet:

- tinycudnn support
- occupancy grid pruning
- hierarchical sampling

These would be good next steps for increasing the PSNR attained by the model and reducing the training time.

## Results

Vanilla NeRF: 29.58 PSNR on test set after ~10hrs training
![Vanilla NeRF](results/vanilla_nerf.gif)





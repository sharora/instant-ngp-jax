import jax.numpy as jnp
from flax import linen as nn


# encoders
class Encoder(nn.Module):
    ...


# sinusoidal basis
class FourierEncoder(Encoder):
    num_freq: int

    @nn.compact
    def __call__(self, x):
        # TODO: perhaps this implementation can be improved
        out = []
        for i in range(self.num_freq):
            out += [
                jnp.sin(x * jnp.pi * 2**i),
                jnp.sin(jnp.pi / 2 - x * jnp.pi * 2**i),
            ]
        return jnp.concatenate([x, jnp.array(out).flatten()])


# TODO: InstantNGP
class HashEncoder(Encoder):
    pass


# TODO: spherical harmonics encoding
# use spherical harmonics function from jax scipy
class SphericalHarmonicsEncoder(Encoder):
    pass

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array, vmap
from jax.scipy.ndimage import map_coordinates
from jax.scipy.special import sph_harm


class Encoder(nn.Module):
    ...


class FourierEncoder(Encoder):
    num_freq: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Array:
        out = []
        for i in range(self.num_freq):
            out += [
                jnp.sin(x * jnp.pi * 2**i),
                jnp.sin(jnp.pi / 2 - x * jnp.pi * 2**i),
            ]
        return jnp.concatenate([x, jnp.array(out).flatten()])


# TODO: can possibly clean this code up a bit more, remove some nested functions
class HashEncoder(Encoder):
    # following notation from the paper exactly
    L: int = 16
    T: int = 2**19
    F: int = 2
    n_min: int = 16
    n_max: int = 4096
    d: int = 3
    pis: jnp.ndarray = jnp.array([1, 2654435761, 805459861])

    def setup(self) -> None:
        # TODO: this is different from paper since the initializer uniformly samples from [0, 1e-4] instead of [-1e-4, 1e-4]
        self.hashtable = self.param(
            "hashtable", nn.initializers.uniform(scale=1e-4), (self.L, self.T, self.F)
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Array:
        assert self.d <= 3, "Only up to 3D is supported."
        assert self.L >= 1, "At least one level is required."
        assert self.n_min > 0, "n_min must be positive."

        # computing resolution levels using n_min, n_max as described in the paper
        b = jnp.exp(jnp.log(self.n_max / self.n_min) / (self.L - 1))
        n_levels = jnp.floor(self.n_min * b ** jnp.arange(self.L))

        def hash_level(x: jnp.ndarray, n_l: Array, hashtable: Array) -> Array:
            # find the lower bound of the voxel
            lb = jnp.floor(x * n_l).astype(jnp.int32)

            # finding all the corners of the voxel
            voxel_indices = jnp.moveaxis(jnp.indices((2,) * self.d), 0, -1)
            voxel_corners = lb + voxel_indices

            def get_feature_from_coords(coords: Array, hashtable: Array) -> Array:
                # hash the coordinates using spatial hash function from the paper
                prods = self.pis[: self.d] * coords
                hash = jax.lax.reduce(prods, 0, jnp.bitwise_xor, (0,)) % self.T
                return hashtable[hash]

            # getting the feature stored at each voxel corner
            voxel_corners = voxel_corners.reshape(-1, self.d)
            features = vmap(get_feature_from_coords, in_axes=(0, None))(
                voxel_corners, hashtable
            )
            features = features.reshape((2,) * self.d + (self.F,))

            # finally use map_coordinates to get the feature at the point (via linear interpolation)
            features = jnp.moveaxis(features, -1, 0)
            interpolated_feature = vmap(map_coordinates, in_axes=(0, None, None))(
                features, list(x * n_l - lb), 1
            )
            return interpolated_feature

        # concatenate the features from each level to get the final feature vector
        return vmap(hash_level, in_axes=(None, 0, 0))(
            x, n_levels, self.hashtable
        ).flatten()


# for encoding ray directions
class SphericalHarmonicsEncoder(Encoder):
    deg: int

    @nn.compact
    def __call__(self, ray: jnp.ndarray) -> Array:
        # get angular coordinates for the ray,, assumes the ray is normalized
        theta = jnp.arctan2(ray[0], ray[1])
        phi = jnp.arccos(ray[2])

        # use function from jax scipy to compute basis
        return sph_harm(jnp.array([self.deg]), jnp.array([self.deg]), theta, phi)


if __name__ == "__main__":
    sptest = SphericalHarmonicsEncoder(4)
    sptest.init(jax.random.PRNGKey(0), jnp.array([1, 1, 1]) / jnp.sqrt(3))

    hash_enc = HashEncoder()
    hash_enc.init(jax.random.PRNGKey(0), jnp.array([1 / 32, 1 / 32, 1 / 32]))

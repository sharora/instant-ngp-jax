import typing as T
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import Array, jit, vmap

from encoders import Encoder, FourierEncoder, HashEncoder, SphericalHarmonicsEncoder


class NerfMLP(nn.Module):
    num_layers_density: int = 8
    num_layers_rgb: int = 2
    density_output_size: int = 256
    layer_width_density: int = 256
    layer_width_rgb: int = 128
    density_skip_layer: T.Optional[int] = num_layers_density // 2

    @nn.compact
    def __call__(
        self, p_enc: jnp.ndarray, view_enc: jnp.ndarray
    ) -> T.Tuple[Array, Array]:
        # first use pos_coords to get the density and features
        x = p_enc
        for i in range(self.num_layers_density):
            x = nn.Dense(self.layer_width_density)(x)
            x = nn.relu(x)
            x = (
                jnp.concatenate([x, p_enc], axis=-1)
                if i == self.density_skip_layer
                else x
            )
        x = nn.Sequential([nn.Dense(self.density_output_size), nn.relu])(x)
        density = x[:, 0]

        # then use the features and view_dir to get the color
        x = jnp.concatenate([x, view_enc], axis=-1)
        rgb_layers = [
            nn.Sequential([nn.Dense(self.layer_width_rgb), nn.relu])
            for _ in range(self.num_layers_rgb - 1)
        ]
        output_layers = [nn.Dense(3), nn.sigmoid]
        rgb = nn.Sequential(rgb_layers + output_layers)(x)
        return density, rgb


# TODO: add background color option, currently assumes it is black
class NeRF(nn.Module):
    position_encoder: Encoder
    direction_encoder: Encoder
    mlp: NerfMLP

    @nn.compact
    def __call__(
        self, ray_origin: jnp.ndarray, ray_direction: jnp.ndarray, rng: jnp.ndarray
    ) -> T.Tuple[Array, Array]:
        # first sample points along every ray
        sample_points, t = sample_points_along_ray(ray_origin, ray_direction, rng, 256)
        ray_dirs = jnp.repeat(
            ray_direction[jnp.newaxis, :], sample_points.shape[0], axis=0
        )

        # encode the points/directions
        point_encodings = vmap(self.position_encoder)(sample_points)
        direction_encodings = vmap(self.direction_encoder)(ray_dirs)

        # get rgb and density at points
        densities, colors = self.mlp(point_encodings, direction_encodings)

        # use the volume rendering equation to get the pixel values for each ray
        weights = rendering_weights(densities, t)

        # compute pixel value and depth
        rgb = jnp.sum(colors * weights[:, jnp.newaxis], axis=0)
        depth = jnp.array([jnp.sum(t * weights, axis=-1)])
        return rgb, depth


@dataclass
class NeRFConfig:
    instant_ngp: bool = False


# TODO: we should probably expose more config options here, works for now
def make_nerf_model(config: NeRFConfig) -> NeRF:
    if config.instant_ngp:
        position_encoder = HashEncoder()
        direction_encoder = SphericalHarmonicsEncoder(deg=4)
        mlp = NerfMLP(
            num_layers_density=1,
            num_layers_rgb=2,
            density_output_size=16,
            layer_width_density=64,
            layer_width_rgb=64,
            density_skip_layer=None,
        )
    else:
        position_encoder = FourierEncoder(num_freq=10)
        direction_encoder = FourierEncoder(num_freq=4)
        mlp = NerfMLP()
    return NeRF(position_encoder, direction_encoder, mlp)


def render_image(
    state: TrainState,
    pose: jnp.ndarray,
    camera_cal: jnp.ndarray,
    image_shape: T.Sequence[int],
    rng: jnp.ndarray,
):
    rays = get_rays(pose, camera_cal, image_shape)
    ray_origin = pose[:3, 3]
    input_rng = jax.random.split(rng, num=rays.shape[:-1])

    # render image row by row (memory is limited)
    @jit
    def render_row(i: int):
        return vmap(state.apply_fn, in_axes=(None, None, 0, 0))(
            {"params": state.params}, ray_origin, rays[i], input_rng[i]
        )

    return jax.lax.map(render_row, jnp.arange(rays.shape[0]))


def rendering_weights(densities: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the weights to multiply each pixel value by along a ray when
    doing volume rendering.
    """
    dists = jnp.concatenate([t[1:] - t[:-1], jnp.array([1e10])])
    alphas = jnp.exp(-dists * densities)
    transmittance = jnp.concatenate([jnp.array([1]), jnp.cumprod(alphas)[:-1]])
    return transmittance * (1 - alphas + 1e-10)


# TODO: switch away from using a hardcoded near/far bound
def sample_points_along_ray(
    ray_origin: jnp.ndarray,
    ray_direction: jnp.ndarray,
    rng: jnp.ndarray,
    num_samples: int,
    near_bound: float = 1.0,
    far_bound: float = 5.0,
):
    interval_size = (far_bound - near_bound) / num_samples
    noise = jax.random.uniform(rng, (num_samples,), maxval=interval_size)
    t = jnp.linspace(near_bound, far_bound, num_samples) + noise
    return ray_origin + ray_direction * t[:, jnp.newaxis], t


def get_rays(
    camera_pose: jnp.ndarray,
    camera_calibration: jnp.ndarray,
    image_shape: T.Sequence[int],
):
    v, u = jnp.indices((image_shape[0], image_shape[1]))
    image_coords = jnp.stack([u + 0.5, v + 0.5, jnp.ones_like(u)], axis=-1)
    return image_coords @ jnp.linalg.inv(camera_calibration).T @ camera_pose[:3, :3].T

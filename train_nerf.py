import typing as T
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import jit, value_and_grad, vmap
from tqdm import tqdm

import wandb
from datasets import SceneTrainDataset, load_scene_train_data
from models import NeRF, NeRFConfig, make_nerf_model, render_image


@dataclass
class TrainConfig:
    epochs: int = 100
    # batch_size: int = 2048
    batch_size: int = 1
    init_lr: float = 5e-5
    lr_decay_rate: float = 0.9
    scene_name: str = "lego"
    checkpoint_dir: str = "./checkpoints"
    nerf_config: NeRFConfig = NeRFConfig()


def create_train_state(nerf: NeRF, config: TrainConfig, rng: jnp.ndarray) -> TrainState:
    """Creates an initial `TrainState`."""
    rng, input_rng = jax.random.split(rng)
    params = nerf.init(rng, jnp.ones(3), jnp.ones(3), input_rng)["params"]
    exponential_decay_scheduler = optax.exponential_decay(
        init_value=config.init_lr,
        transition_steps=0,
        decay_rate=config.lr_decay_rate,
        transition_begin=0,
        staircase=False,
    )
    tx = optax.adam(learning_rate=exponential_decay_scheduler)
    return TrainState.create(apply_fn=nerf.apply, params=params, tx=tx)


@jit
def train_step(
    state, rays, origins, pixels: jnp.ndarray, rng: jnp.ndarray
) -> T.Tuple[TrainState, float]:
    """Evaluates the model on a batch of data."""

    def loss_fn(params):
        input_rng = jax.random.split(rng, num=rays.shape[0])
        rgb, _ = vmap(state.apply_fn, in_axes=(None, 0, 0, 0))(
            {"params": params}, origins, rays, input_rng
        )
        pixel_preds = rgb
        return jnp.mean((pixel_preds - pixels) ** 2)

    loss, grads = value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_epoch(
    state: TrainState, train_ds: SceneTrainDataset, batch_size: int, rng: jnp.ndarray
) -> TrainState:
    """Train for a single epoch."""
    train_ds_size = len(train_ds.train_rays)
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in tqdm(perms):
        rays = train_ds.train_rays[perm]
        origins = train_ds.train_origins[perm]
        pixels = train_ds.train_pixels[perm]
        rng, input_rng = jax.random.split(rng)
        state, loss = train_step(state, rays, origins, pixels, input_rng)
        wandb.log({"loss": loss})
    return state


def validate_model(
    state: TrainState, train_ds: SceneTrainDataset, rng: jnp.ndarray
) -> None:
    val_mses: T.List[jax.Array] = []
    val_images: T.List[wandb.Image] = []
    val_depths: T.List[wandb.Image] = []
    for pose, img in zip(train_ds.validation_poses, train_ds.validation_imgs):
        rng, input_rng = jax.random.split(rng)
        rgb, depth = render_image(
            state,
            pose,
            train_ds.camera_calibration,
            img.shape,
            input_rng,
        )
        val_mses.append(jnp.mean((rgb - img) ** 2))
        val_images.append(wandb.Image(jnp.clip(rgb, 0, 1)))
        # TODO: maybe there is a better solution for normalizing the depth image
        val_depths.append(wandb.Image(jnp.clip(depth / jnp.max(depth), 0, 1)))
    val_mse = np.mean(val_mses)
    wandb.log(
        {
            "validation_mse": val_mse,
            "validation_psnr": -10 * jnp.log10(val_mse),
            "validation_rgb": val_images,
            "validation_depth": val_depths,
        }
    )


def train_model(
    config: TrainConfig,
    train_ds: SceneTrainDataset,
    state: TrainState,
    rng: jnp.ndarray,
) -> TrainState:
    for epoch in range(config.epochs):
        print(f"Epoch {epoch}")

        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, config.batch_size, input_rng)

        rng, input_rng = jax.random.split(rng)
        validate_model(state, train_ds, input_rng)

        checkpoints.save_checkpoint(
            ckpt_dir=config.checkpoint_dir, target=state, step=epoch
        )
    return state


def main(config: TrainConfig) -> None:
    # starting wandb
    wandb.login()
    wandb.init(project="nerf", config=config.__dict__)

    # load data
    train_data = load_scene_train_data(config.scene_name)

    # create initial state
    # TODO: add option to resume training
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    nerf = make_nerf_model(config.nerf_config)
    init_state = create_train_state(nerf, config, init_rng)

    # train model
    output_state = train_model(config, train_data, init_state, rng)

    # TODO: maybe uploading the model to wandb


if __name__ == "__main__":
    config = TrainConfig()
    main(config)

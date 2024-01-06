import os
import typing as T
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jaxlie import SE3, SO3
from PIL import Image
from tqdm import tqdm

from datasets import SceneTestDataset, load_test_dataset
from models import make_nerf_model, render_image
from train_nerf import TrainConfig, create_train_state


def save_images_as_gif(images: T.List[np.ndarray], out_path: str) -> None:
    pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
    pil_images[0].save(
        Path(out_path).with_suffix(".gif"),
        save_all=True,
        append_images=pil_images[1:],
        duration=50,
        loop=0,
    )


# TODO: not confident that this function will work when the model config changes (e.g. number of layers, hidden size, etc.)
def restore_checkpoint(checkpoint_path: str) -> TrainState:
    # create dummy state for restoring the checkpoint
    rng = jax.random.PRNGKey(0)
    config = TrainConfig()
    target_state = create_train_state(make_nerf_model(config.nerf_config), config, rng)

    # restore the checkpoint from given path
    return checkpoints.restore_checkpoint(
        ckpt_dir=os.path.abspath(checkpoint_path), target=target_state
    )


def eval_model(
    state: TrainState,
    test_dataset: SceneTestDataset,
    rng: jnp.ndarray,
    video_out_path: T.Optional[str] = None,
) -> None:
    images: T.List[np.ndarray] = []
    psnrs: T.List[jax.Array] = []
    for i, pose in tqdm(enumerate(test_dataset.test_poses)):
        rng, input_rng = jax.random.split(rng)
        rgb, _ = render_image(
            state,
            pose,
            test_dataset.camera_calibration,
            test_dataset.test_imgs[0].shape,
            input_rng,
        )
        images.append(np.array(jnp.clip(rgb, 0, 1)))

        # compute metrics
        mse = jnp.mean((rgb - test_dataset.test_imgs[i]) ** 2)
        psnr = -10 * jnp.log10(mse)
        psnrs.append(psnr)

    print(f"Average PSNR on Test Set: {jnp.mean(jnp.array(psnrs))}")
    if video_out_path is not None:
        save_images_as_gif(images, video_out_path)


def main(
    scene_name: str,
    checkpoint_path: str,
    video_out_path: T.Optional[str] = None,
) -> None:
    # loads test dataset, restores checkpoint, and evaluates the model
    test_dataset = load_test_dataset(scene_name)
    saved_state = restore_checkpoint(checkpoint_path)
    eval_model(saved_state, test_dataset, jax.random.PRNGKey(0), video_out_path)


if __name__ == "__main__":
    tyro.cli(main)

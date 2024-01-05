import functools
import json
import os
import typing as T
import zipfile
from dataclasses import dataclass

import gdown
import jax.numpy as jnp
import jaxlie
from jax import vmap
from PIL import Image

from models import get_rays

DATASET_FOLDER = "nerf_synthetic"
DATASET_URL = "https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=drive_link"


@dataclass
class SceneTrainDataset:
    train_rays: jnp.ndarray
    train_origins: jnp.ndarray
    train_pixels: jnp.ndarray
    validation_imgs: T.List[jnp.ndarray]
    validation_poses: T.List[jnp.ndarray]
    camera_calibration: jnp.ndarray


@dataclass
class SceneTestDataset:
    test_poses: T.List[jnp.ndarray]
    test_imgs: T.Optional[T.List[jnp.ndarray]] = None


def download_dataset() -> None:
    """
    Downloads and extracts the NeRF synthetic dataset.
    """
    try:
        output = DATASET_FOLDER + ".zip"
        gdown.download(DATASET_URL, output=output, quiet=True, fuzzy=True)

        # unzip
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(DATASET_FOLDER)
        os.remove(output)
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")


def get_raw_scene_data(
    scene_name: str, set: str
) -> T.Tuple[T.List[jnp.ndarray], T.List[jnp.ndarray], jnp.ndarray]:
    if not os.path.exists(DATASET_FOLDER):
        download_dataset()
    scene_root = os.path.join(DATASET_FOLDER, DATASET_FOLDER, scene_name)

    # load set data
    set_transforms_file = "transforms_{}.json".format(set)
    with open(os.path.join(scene_root, set_transforms_file), "r") as f:
        set_transforms = json.load(f)

    # load images/poses
    # convert blender camera coordinate system to RDF
    blender_cam_T_rdf = jaxlie.SE3.from_rotation(
        jaxlie.SO3.from_x_radians(jnp.pi)
    ).as_matrix()
    images, poses = [], []
    for frame in set_transforms["frames"]:
        file_path, pose = frame["file_path"], frame["transform_matrix"]
        image_path = os.path.join(scene_root, file_path + ".png")
        image = jnp.array(Image.open(image_path))[:, :, :3] / 255.0
        pose = jnp.array(pose) @ blender_cam_T_rdf
        images.append(image)
        poses.append(pose)

    # construct camera intrinsics matrix
    x_fov = set_transforms["camera_angle_x"]
    cx = images[0].shape[1] / 2
    cy = images[0].shape[0] / 2
    f = cx / jnp.tan(x_fov / 2)
    K = jnp.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    return images, poses, K


def load_scene_train_data(
    scene_name: str, num_val_images: int = 5
) -> SceneTrainDataset:
    # first load train data
    images, poses, camera_cal = get_raw_scene_data(scene_name, "train")

    # then process it into rays, origins, pixels
    ray_func = functools.partial(get_rays, image_shape=images[0].shape)
    train_rays = vmap(ray_func, in_axes=(0, None))(jnp.array(poses), camera_cal)
    train_rays = train_rays.reshape((-1, 3))
    train_rays = train_rays / jnp.linalg.norm(train_rays, axis=-1, keepdims=True)
    train_pixels = jnp.array(images).reshape((-1, 3))
    train_origins = jnp.repeat(
        jnp.array(poses)[:, :3, 3], images[0].shape[0] * images[0].shape[1], axis=0
    )
    assert train_rays.shape == train_pixels.shape == train_origins.shape

    # next load validation data
    val_images, val_poses, _ = get_raw_scene_data(scene_name, "val")
    val_images = val_images[:num_val_images]
    val_poses = val_poses[:num_val_images]

    # lastly package it into a SceneTrainDataset
    return SceneTrainDataset(
        train_rays, train_origins, train_pixels, val_images, val_poses, camera_cal
    )


def plot_scene_train_data(dataset: SceneTrainDataset) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)
    # NOTE: assumes all images are the same size (including validation)
    img_shape = dataset.validation_imgs[0].shape
    train_rays = dataset.train_rays.reshape((-1, *img_shape))
    train_origins = dataset.train_origins.reshape((-1, *img_shape))
    arrows = [
        (0, 0, "red"),
        (0, img_shape[1], "green"),
        (img_shape[0], 0, "blue"),
        (img_shape[0], img_shape[1], "yellow"),
    ]
    for i in range(train_rays.shape[0]):
        # TODO: plotting frustums might look better but this works for now
        # should generally show with way the camera is pointing + colors give orientation
        for x, y, color in arrows:
            ax.quiver(
                *train_origins[i, y, x],
                *train_rays[i, y, x],
                color=color,
                arrow_length_ratio=0.1,
            )
    fig.add_axes(ax)
    plt.show()


if __name__ == "__main__":
    dataset = load_scene_train_data("lego")
    plot_scene_train_data(dataset)

import tensorflow as tf
import os
from glob import glob
import math
import numpy as np

MEAN = 0.47194585
STD = 0.16105755


def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(x, -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return tf.squeeze(y, axis=-1)


def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6., 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


def _prepare_directory(model_directory, n_folds=5):
    """

    :param model_directory:
    :param n_folds:
    :return:
    """
    try:
        os.makedirs(f'{model_directory}/{tf.estimator.ModeKeys.TRAIN}/images', exist_ok=True)
        os.makedirs(f'{model_directory}/{tf.estimator.ModeKeys.EVAL}/images', exist_ok=True)
    except Exception as e:
        print(e)

    try:
        os.makedirs(f'{model_directory}/{tf.estimator.ModeKeys.TRAIN}/masks', exist_ok=True)
        os.makedirs(f'{model_directory}/{tf.estimator.ModeKeys.EVAL}/masks', exist_ok=True)

    except Exception as e:
        print(e)

    for fold in range(n_folds):
        try:
            os.mkdir(f'{model_directory}/{tf.estimator.ModeKeys.TRAIN}/images/fold{fold}')
        except Exception as e:
            print(e)
            [os.remove(x) for x in glob(f'{model_directory}/{tf.estimator.ModeKeys.TRAIN}/images/fold{fold}/*')]

        try:
            os.mkdir(f'{model_directory}/{tf.estimator.ModeKeys.TRAIN}/masks/fold{fold}')
        except Exception as e:
            print(e)
            [os.remove(x) for x in glob(f'{model_directory}/{tf.estimator.ModeKeys.TRAIN}/masks/fold{fold}/*')]

        try:
            os.mkdir(f'{model_directory}/{tf.estimator.ModeKeys.EVAL}/images/fold{fold}')
        except Exception as e:
            print(e)
            [os.remove(x) for x in glob(f'{model_directory}/{tf.estimator.ModeKeys.EVAL}/images/fold{fold}/*')]

        try:
            os.mkdir(f'{model_directory}/{tf.estimator.ModeKeys.EVAL}/masks/fold{fold}')
        except Exception as e:
            print(e)
            [os.remove(x) for x in glob(f'{model_directory}/{tf.estimator.ModeKeys.EVAL}/masks/fold{fold}/*')]


def create_symlinks(data_dir, model_dir, mode, idx, fold):
    if len(glob(f"{model_dir}/{mode}/images/*.png")) == 0:
        print(len(glob(f"{model_dir}/{mode}/images/*.png")))
        for x in idx:
            os.symlink(f'{data_dir}/images/{x}.png',
                       f'{model_dir}/{mode}/images/fold{fold}/{x}.png')
            os.symlink(f'{data_dir}/masks/{x}.png',
                       f'{model_dir}/{mode}/masks/fold{fold}/{x}.png')
    else:
        print(f"Fold has already been processed, continuing with the same {mode} set")


def _parse_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.cast(tf.image.decode_jpeg(image_string,
                                                 channels=1),
                            tf.float32) / 255.
    image_decoded.set_shape([101, 101, 1])  # tell tf the shape of the image
    return image_decoded


def read_image(X, y):
    return {'images': _parse_image(X)}, _parse_image(y)


def read_and_preprocess(X,
                        y,
                        augment=False,
                        horizontal_flip=True,
                        vertical_flip=True,
                        rotate_range=10,
                        crop_probability=.5,
                        crop_min_percent=0.9,
                        crop_max_percent=1.1,
                        height_shift_range=0.2,
                        width_shift_range=0.2,
                        brightness_range=0.0):
    """
    Image augmentation like Keras but pure tensorflow implementation

    :param X:
    :param y:
    :param augment:
    :param horizontal_flip:
    :param vertical_flip:
    :param rotate_range:
    :param crop_probability:
    :param crop_min_percent:
    :param crop_max_percent:
    :param brightness_range:
    :param height_shift_range:
    :param width_shift_range:

    :return:
    """

    image = _parse_image(X)
    mask = _parse_image(y)

    image = (image - MEAN) / STD

    if augment:
        # add some padding to image for rotations and random cropping
        image = tf.pad(image, tf.constant([[40, 40], [40, 40], [0, 0]]), mode='REFLECT')
        mask = tf.pad(mask, tf.constant([[40, 40], [40, 40], [0, 0]]), mode='REFLECT')

        # from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
        shp = tf.shape(image)
        # batch size is 1 because we map this function to each individual entry
        batch_size, height, width = 1, shp[0], shp[1]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)

        image, mask = tf.cond(tf.greater(tf.random_uniform((), 0, 1.0), 0.5),
                              lambda: (tf.image.transpose_image(image), tf.image.transpose_image(mask)),
                              lambda: (image, mask))

        if brightness_range > 0:
            image = tf.image.random_brightness(image=image, max_delta=brightness_range)

        if horizontal_flip:
            coin = tf.less(tf.random_uniform((), 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        angle_rad = rotate_range / 180 * math.pi
        angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
        transforms.append(
            tf.contrib.image.angles_to_projective_transforms(
                angles, height, width))

        if width_shift_range:
            tx = np.random.uniform(-width_shift_range, width_shift_range) * height
        else:
            tx = 0
        if height_shift_range:
            ty = np.random.uniform(-height_shift_range, height_shift_range) * height
        else:
            ty = 0

        transforms.append(tf.contrib.image.matrices_to_flat_transforms(
            tf.convert_to_tensor(
                [[1, 0, tx],
                 [0, 1, ty],
                 [0, 0, 1]],
                dtype=tf.float32)
        ))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct, tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]), tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            image = tf.contrib.image.transform(
                image,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')
            mask = tf.contrib.image.transform(
                mask,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='NEAREST')

        image = tf.image.central_crop(image, 101 / 181)
        mask = tf.image.central_crop(mask, 101 / 181)

    image = tf.concat([image, laplace(image)], axis=-1)
    # image = laplace(image)

    return {'images': image}, mask


def single_transformation(X, transformation):
    """
    Single transformation on an image
    :param X:
    :param y:
    :param transformation:
    :return:
    """

    if transformation == "vertical":
        image = tf.image.flip_up_down(image)

    elif transformation == "horizontal":
        image = tf.image.flip_left_right(image)

    elif transformation == "transpose":
        image = tf.image.transpose_image(image)

    elif transformation == "none":
        image = tf.identity(image)

    else:
        raise ValueError(f"Unknown transformation {transformation}")

    return {'images': image}

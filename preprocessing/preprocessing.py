import tensorflow as tf
import os
from glob import glob
import math


def _prepare_directory(model_directory, n_folds=5):
    """

    :param model_directory:
    :param n_folds:
    :return:
    """
    try:
        os.makedirs(f'{model_directory}/train/images', exist_ok=True)
    except Exception as e:
        print(e)

    try:
        os.makedirs(f'{model_directory}/train/masks', exist_ok=True)
    except Exception as e:
        print(e)

    for fold in range(n_folds):
        try:
            os.mkdir(f'{model_directory}/train/images/fold{fold}')
            os.mkdir(f'{model_directory}/train/masks/fold{fold}')

        except Exception as e:
            print(e)
            print(f'{model_directory}/train/images/fold{fold} already exists!')
            [os.remove(x) for x in glob(f'{model_directory}/train/images/fold{fold}/*')]
            [os.remove(x) for x in glob(f'{model_directory}/train/masks/fold{fold}/*')]


def _parse_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.cast(tf.image.decode_jpeg(image_string,
                                                 channels=1),
                            tf.float32) / 255.
    image_decoded.set_shape([101, 101, 1])  # tell tf the shape of the image
    return image_decoded


# def _parse_mask(filename):
#     image_string = tf.read_file(filename)
#     image_decoded = tf.cast(tf.image.decode_jpeg(image_string, channels=1).set_shape(101, 101, 1), tf.float32) / 255.
#     image_decoded = tf.reshape(image_decoded, (101, 101, 1))
#     return image_decoded


def read_image(X, y):
    return {'images': _parse_image(X)}, _parse_image(y)


def read_and_preprocess(X, y, horizontal_flip=True,
                        vertical_flip=True, rotate_range=20,
                        crop_probability=.5,
                        crop_min_percent=0.8,
                        crop_max_percent=1.0):
    """
    Image augmentation like Keras but pure tensor implementation

    :param X:
    :param y:
    :param horizontal_flip:
    :param vertical_flip:
    :param rotate_range:
    :param crop_probability:
    :param crop_min_percent:
    :param crop_max_percent:
    :return:
    """

    image = _parse_image(X)
    mask = _parse_image(y)

    image = tf.pad(image, tf.constant([[20, 20], [20, 20], [0, 0]]), mode='REFLECT')
    mask = tf.pad(mask, tf.constant([[20, 20], [20, 20], [0, 0]]), mode='REFLECT')

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

    if crop_probability > 0:
        crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                     crop_max_percent)
        left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
        top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
        crop_transform = tf.stack([
            crop_pct,
            tf.zeros([batch_size]), top,
            tf.zeros([batch_size]), crop_pct, left,
            tf.zeros([batch_size]),
            tf.zeros([batch_size])
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

    image = tf.image.central_crop(image, 101 / 141)
    mask = tf.image.central_crop(mask, 101 / 141)

    return {'images': image}, mask

import tensorflow as tf
from utils import get_available_gpus
from core.layers import split_separable_conv2d, _upsample
from core.resnet import resnet_v2_34_beta, bottleneck
from tensorflow.contrib import layers as layers_lib
from core.losses import lovasz_loss
from core.metric import mIOU, mean_accuracy
from preprocessing.preprocessing import _prepare_directory, read_and_preprocess, read_image
from sklearn.model_selection import StratifiedKFold
import os

slim = tf.contrib.slim

# l2 regularisation param
WEIGHT_DECAY = 0.001
# batch norm params
BATCH_NORM_DECAY = 0.99
BATCH_NORM_EPSILON = 0.001
BATCH_NORM_SCALE = True
# resnet output stride
OUTPUT_STRIDE = 8
# image shape
INPUT_SHAPE = (101, 101)

BASE_DEPTH = 256

DATA_FORMAT = "NCHW"


class Model:

    def __init__(self, model_dir, data_directory, data_format=DATA_FORMAT,
                 lr=0.001, n_gpus=2, n_fold=5, seed=42):
        """
        High level wrapper to perform multi GPU training with tf.contrib.distribute.MirroredStrategy
        and tf.Estimator.
        The base models are built using tf.slim.
        All the data is processed using tf.data.Dataset and tf.image.
        The preprocessing currently runs on the CPU (optimal?)
        Additionally built ResNet with NCHW format support for (potentially) faster GPU and MKL optimised CPU operations


        :param model_dir:
        :param data_directory:
        :param data_format:
        :param lr:
        :param n_gpus:
        :param n_fold:
        :param seed:
        """

        if data_format in ["NCHW", "NHWC"]:
            self.data_format = data_format
        else:
            raise ValueError(f"Unknown data format {data_format}. Has to be either NCHW or NHWC")

        # Pathing stuff
        self.model_name = model_dir.split('/')[-1]
        self.model_dir = model_dir
        self.data_dir = data_directory

        # Estimator stuff
        available_gpus = get_available_gpus()
        distribution = tf.contrib.distribute.MirroredStrategy(
            devices=available_gpus[:n_gpus])
        self.config = tf.estimator.RunConfig(
            train_distribute=distribution)

        # Additional args to fine tune training at high level
        self.data_format = data_format
        self.n_gpus = n_gpus
        self.n_folds = n_fold
        self.seed = seed
        self.lr = lr

        _prepare_directory(self.model_dir, self.n_folds)

        self.skf = StratifiedKFold(n_splits=self.n_folds,
                                   shuffle=True,
                                   random_state=self.seed)

    def train(self, X, y, batch_size, steps=100):
        train_idx = []
        test_idx = []
        i = 0
        for train_index, test_index in self.skf.split(X, y):
            train_idx.append(train_index.tolist())
            test_idx.append(test_index.tolist())

        if batch_size % self.n_gpus == 0:
            batch_size = int(batch_size / self.n_gpus)
        else:
            raise ValueError("Batch size must be a multiple of n_gpus")

        for train_index, test_index in zip(train_idx, test_idx):
            tf.logging.info(f"Processing fold {i+1}")

            model = tf.estimator.Estimator(
                model_fn=self.build_model_fn_optimizer(i),
                model_dir=f"{self.model_dir}/fold{i}",
                config=self.config)

            train_spec = tf.estimator.TrainSpec(
                input_fn=self.make_input_fn(X, tf.estimator.ModeKeys.TRAIN, train_index, i, batch_size),
                max_steps=steps
            )

            eval_spec = tf.estimator.EvalSpec(
                input_fn=self.make_input_fn(X, tf.estimator.ModeKeys.EVAL, test_index, i, batch_size * 2),
                steps=len(test_index) // (batch_size * 2),
                throttle_secs=120,
                start_delay_secs=120,
            )

            tf.estimator.train_and_evaluate(
                model,
                train_spec,
                eval_spec
            )

            tf.logging.info(f'Finished training fold {i+1}.')

            i += 1

    def evaluate(self):
        pass

    def make_input_fn(self, X, mode, index, fold, batch_size):

        for x in X[index]:
            os.symlink(f'{self.data_dir}/images/{x}.png',
                       f'{self.model_dir}/train/images/fold{fold}/{x}.png')
            os.symlink(f'{self.data_dir}/masks/{x}.png',
                       f'{self.model_dir}/train/masks/fold{fold}/{x}.png')

        if mode == tf.estimator.ModeKeys.TRAIN:

            def train_input_fn():
                # A vector of filenames.
                filenames = tf.constant(
                    [f'{self.model_dir}/train/images/fold{fold}/{x}.png' for x in X[index]])

                # `labels[i]` is the label for the image in `filenames[i].
                labels = tf.constant(
                    [f'{self.model_dir}/train/masks/fold{fold}/{x}.png' for x in X[index]])

                dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

                dataset = dataset.map(read_and_preprocess)

                dataset = dataset.shuffle(batch_size * 10)
                dataset = dataset.repeat()
                dataset = dataset.batch(batch_size)
                dataset = dataset.prefetch(2)

                return dataset

        elif mode == tf.estimator.ModeKeys.EVAL:
            def train_input_fn():
                # A vector of filenames.
                filenames = tf.constant(
                    [f'{self.model_dir}/train/images/fold{fold}/{x}.png' for x in X[index]])

                # `labels[i]` is the label for the image in `filenames[i].
                labels = tf.constant(
                    [f'{self.model_dir}/train/masks/fold{fold}/{x}.png' for x in X[index]])

                dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

                dataset = dataset.map(read_image)

                dataset = dataset.batch(batch_size)
                dataset = dataset.prefetch(2)

                return dataset
        else:
            raise ValueError(f"Unknown mode {mode}")

        return train_input_fn

    def build_model_fn_optimizer(self, fold):

        def _model_fn(features, labels, mode):

            if mode == tf.estimator.ModeKeys.TRAIN:
                is_training = True
            else:
                is_training = False

            with tf.variable_scope(self.model_name):

                if self.data_format == "NCHW":
                    input = tf.transpose(features["images"], [0, 3, 1, 2], name="image_input")
                    labels = tf.transpose(labels, [0, 3, 1, 2], name="label_input")
                else:
                    input = tf.identity(features["images"], name="image_input")
                    labels = tf.identity(labels, name="label_input")

                with slim.arg_scope(slim.nets.resnet_v2.resnet_arg_scope(weight_decay=WEIGHT_DECAY,
                                                                         batch_norm_decay=BATCH_NORM_DECAY,
                                                                         batch_norm_epsilon=BATCH_NORM_EPSILON,
                                                                         batch_norm_scale=BATCH_NORM_SCALE)):
                    with slim.arg_scope([slim.conv2d,
                                         bottleneck,
                                         layers_lib.max_pool2d,
                                         slim.batch_norm,
                                         slim.separable_conv2d,
                                         _upsample], data_format=self.data_format):
                        with slim.arg_scope([slim.batch_norm], is_training=is_training):

                            net, end_points = resnet_v2_34_beta(input,
                                                                is_training=is_training,
                                                                global_pool=False,
                                                                output_stride=OUTPUT_STRIDE,
                                                                multi_grid=(2, 2, 2),
                                                                scope="resnet_v2")

                            with tf.variable_scope("assp"):
                                atrous_output = end_points[f'{self.model_name}/resnet_v2/block4']

                                if self.data_format == "NCHW":
                                    output_size = atrous_output.get_shape().as_list()[2:4]
                                else:
                                    output_size = atrous_output.get_shape().as_list()[1:3]

                                with tf.variable_scope("conv"):
                                    assp_1 = slim.conv2d(atrous_output, num_outputs=BASE_DEPTH, kernel_size=1,
                                                         scope='conv_1x1')
                                    assp_2 = split_separable_conv2d(atrous_output, filters=BASE_DEPTH, kernel_size=3,
                                                                    rate=2, scope='conv_3x3_1')
                                    assp_3 = split_separable_conv2d(atrous_output, filters=BASE_DEPTH, kernel_size=3,
                                                                    rate=4, scope='conv_3x3_2')
                                    assp_4 = split_separable_conv2d(atrous_output, filters=BASE_DEPTH, kernel_size=3,
                                                                    rate=8, scope='conv_3x3_3')

                                with tf.variable_scope("pooling"):
                                    if self.data_format == "NCHW":
                                        axis = [2, 3]
                                    else:
                                        axis = [1, 2]

                                    assp_5 = tf.reduce_mean(atrous_output, axis=axis, keepdims=True,
                                                            name='mean_pooling')
                                    assp_5 = slim.conv2d(assp_5, num_outputs=BASE_DEPTH, kernel_size=1,
                                                         scope='conv_1x1')
                                    assp_5 = _upsample(assp_5, out_shape=output_size)

                                assp_concat = tf.concat([assp_1, assp_2, assp_3, assp_4, assp_5], axis=-1)
                                assp_output = slim.conv2d(assp_concat, num_outputs=BASE_DEPTH, kernel_size=1,
                                                          scope='conv_1x1')

                            assp_upsample = _upsample(assp_output, out_shape=(26, 26))

                            with tf.variable_scope("decoder"):
                                atrous_output = end_points[
                                    f'{self.model_name}/resnet_v2/block1/unit_1/bottleneck_v2/conv3']

                                decoder = slim.conv2d(atrous_output, num_outputs=BASE_DEPTH, kernel_size=1,
                                                      scope='conv_1x1')
                                decoder = tf.concat([decoder, assp_upsample], axis=-1, name='concat_assp')

                                decoder = slim.conv2d(decoder, num_outputs=1, kernel_size=3, activation_fn=None,
                                                      scope='conv_3x3')

                                preactivation_output = _upsample(decoder, out_shape=INPUT_SHAPE)
                                output = tf.nn.sigmoid(preactivation_output, name='sigmoid_output')
                                if self.data_format == "NCHW":
                                    predicted = tf.to_float(tf.expand_dims(
                                        tf.argmax(output,
                                                  axis=1,
                                                  output_type=tf.int32),
                                        1))
                                else:
                                    predicted = tf.to_float(tf.expand_dims(
                                        tf.argmax(output,
                                                  axis=3,
                                                  output_type=tf.int32),
                                        3))

            if mode == tf.estimator.ModeKeys.PREDICT:
                loss = None
                train_op = None
                evalmetrics = None
                # summary_hook = None

            elif mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN):

                def loss_fn():
                    with tf.device("/device:CPU:0"):
                        # run loss on CPU
                        return lovasz_loss(labels, output, data_format=self.data_format)

                loss = loss_fn()

                iou = mIOU(labels, output, name="mean_iou_metric")
                acc = mean_accuracy(labels, predicted, name="mean_acc_metric")

                evalmetrics = {"mean_iou": iou,
                               "mean_acc": acc}

                with tf.variable_scope("metrics"):
                    tf.summary.scalar('mean_acc', acc[1])
                    tf.summary.scalar('mean_iou', iou[1])

                # summary_hook = tf.train.SummarySaverHook(
                #     100,
                #     output_dir=os.path.join(self.directory, self.model_name, str(fold)),
                #     summary_op=tf.summary.merge_all())

                if mode == tf.estimator.ModeKeys.TRAIN:

                    lr = tf.train.exponential_decay(self.lr, tf.train.get_or_create_global_step(),
                                                    10000, 0.5, staircase=False,
                                                    name='learning_rate')
                    optimizer = tf.train.AdamOptimizer(lr)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                else:
                    train_op = None

            else:
                raise ValueError(f"Unknown mode {mode}")

            estimator = tf.estimator.EstimatorSpec(
                mode,
                predictions={"probabilities": output,
                             "class": predicted},
                loss=loss,
                train_op=train_op,
                eval_metric_ops=evalmetrics,
                # training_hooks=[summary_hook],
            )

            return estimator

        return _model_fn

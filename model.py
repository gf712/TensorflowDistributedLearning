import tensorflow as tf
from utils import get_available_gpus
from core.resnet import resnet_model
from core.losses import lovasz_loss
from core.metric import mIOU, mean_accuracy
from preprocessing.preprocessing import _prepare_directory, read_and_preprocess, single_transformation, \
    create_symlinks
from sklearn.model_selection import StratifiedKFold
import os
import functools

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
        Currently NCHW format support is experimental and the speed up is minor (about 5-10%)


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
            save_checkpoints_steps=500,
            train_distribute=distribution,
            save_summary_steps=0  # running summary manually
        )

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
        """
        Train defined model using tf.estimator.train_and_evaluate

        :param X:
        :param y:
        :param batch_size:
        :param steps:
        :return:
        """

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
            tf.logging.info(f"Processing fold {i}")

            model = tf.estimator.Estimator(
                model_fn=self.build_model_fn_optimizer(),
                model_dir=f"{self.model_dir}/fold{i}",
                config=self.config,
                params={'fold': i,
                        'threshold': 0.5  # probability threshold for positive classification
                        }
            )

            create_symlinks(self.data_dir, self.model_dir, tf.estimator.ModeKeys.TRAIN, X[train_index], i)

            # make_input_fn(self, mode, fold, batch_size, augment, shuffle)
            train_spec = tf.estimator.TrainSpec(
                input_fn=self._make_input_fn(mode=tf.estimator.ModeKeys.TRAIN,
                                             fold=i,
                                             batch_size=batch_size,
                                             augment=True,
                                             shuffle=True),
                max_steps=steps
            )

            create_symlinks(self.data_dir, self.model_dir, tf.estimator.ModeKeys.EVAL, X[test_index], i)

            # for inference increasing the batch size to double should not cause any issues
            eval_spec = tf.estimator.EvalSpec(
                input_fn=self._make_input_fn(mode=tf.estimator.ModeKeys.EVAL,
                                             fold=i,
                                             batch_size=batch_size * 2,
                                             augment=False,
                                             shuffle=False),
                steps=len(test_index) // (batch_size * 2),
                throttle_secs=120,
                start_delay_secs=120,
            )

            tf.estimator.train_and_evaluate(
                model,
                train_spec,
                eval_spec
            )

            tf.logging.info(f'Finished training fold {i}.')

            i += 1

    def predict(self, test_dir, batch_size, tti=False):
        """
        Runs evaluation with or without test time augmentation.
        Returns a probability, which in case of tti=True is the average of each augmentation

        :return:
        """

        for i in range(self.n_folds):

            if tti:
                transformations = ["none"]
            else:
                transformations = ["vertical", "horizontal", "transpose", "none"]

            for transformation in transformations:
                input_fn = self._make_test_input(batch_size, test_dir, transformation)

                model = tf.estimator.Estimator(
                    model_fn=self.build_model_fn_optimizer(),
                    model_dir=f"{self.model_dir}/fold{i}",
                    config=self.config,
                    params={'fold': i,
                            'transformation': transformation}
                )

                result_iterator = model.predict(input_fn=input_fn)

    def _make_test_input(self, batch_size, test_directory, tti):

        """

        :param batch_size:
        :param tti:
        :return:
        """

        def test_input_fn():
            # A vector of filenames.
            filenames = tf.constant(
                tf.gfile.Glob(f'{test_directory}/*.png'))

            dataset = tf.data.Dataset.from_tensor_slices(filenames)

            dataset = dataset.map(functools.partial(single_transformation, {'transformation': tti}))

            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(self.n_gpus * 2)  # prefetch twice as many batches as there are GPUs

            return dataset

        return test_input_fn

    def _make_input_fn(self, mode, fold, batch_size, augment, shuffle):

        tf.logging.info(f'{self.model_dir}/{mode}/images/fold{fold}/*.png')

        def input_fn():
            # A vector of filenames.
            filenames = tf.constant(
                tf.gfile.Glob(f'{self.model_dir}/{mode}/images/fold{fold}/*.png'))

            # `labels[i]` is the label for the image in `filenames[i].
            labels = tf.constant(
                tf.gfile.Glob(f'{self.model_dir}/{mode}/masks/fold{fold}/*.png'))

            dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

            dataset = dataset.map(functools.partial(read_and_preprocess, augment=augment))

            if shuffle:
                dataset = dataset.shuffle(batch_size * 10)

            dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(self.n_gpus * 2)  # prefetch twice as many batches as there are GPUs

            return dataset

        return input_fn

    def build_model_fn_optimizer(self):

        def _model_fn(features, labels, mode, params):

            fold = params["fold"]
            if "threshold" in params:
                threshold = params["threshold"]
            else:
                threshold = 0.5

            if mode == tf.estimator.ModeKeys.TRAIN:
                is_training = True
            else:
                is_training = False

            with tf.variable_scope(self.model_name):

                if self.data_format == "NCHW":
                    input = tf.transpose(features["images"], [0, 3, 1, 2], name="image_input")
                    if not tf.estimator.ModeKeys.EVAL:
                        labels = tf.transpose(labels, [0, 3, 1, 2], name="label_input")
                else:
                    input = tf.identity(features["images"], name="image_input")
                    if not tf.estimator.ModeKeys.EVAL:
                        labels = tf.identity(labels, name="label_input")

                preactivation_output = resnet_model(
                    weight_decay=WEIGHT_DECAY,
                    batch_norm_decay=BATCH_NORM_DECAY,
                    batch_norm_epsilon=BATCH_NORM_EPSILON,
                    batch_norm_scale=BATCH_NORM_SCALE,
                    data_format=self.data_format,
                    is_training=is_training,
                    output_stride=OUTPUT_STRIDE,
                    base_depth=BASE_DEPTH,
                    input_shape=INPUT_SHAPE
                )
                
                output = tf.nn.sigmoid(preactivation_output, name='sigmoid_output')
                predicted = tf.to_float(tf.greater(output, threshold))

            eval_hook = []
            training_hook = []

            if mode == tf.estimator.ModeKeys.PREDICT:
                loss = None
                train_op = None
                evalmetrics = None

                # reverse test time augmentation -> this works for all flipping of images along a given axis
                # transpose, horizontal and vertical flip (and no augmentation)
                output = single_transformation(predicted, params["transformation"])

            elif mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN):

                def loss_fn():
                    with tf.device("/device:CPU:0"):
                        # run loss on CPU
                        return lovasz_loss(labels,
                                           preactivation_output,
                                           data_format=self.data_format)

                loss = loss_fn()

                iou = mIOU(labels, predicted, name="mean_iou_metric")
                acc = mean_accuracy(labels, predicted, name="mean_acc_metric")

                evalmetrics = {"metrics/mean_iou": iou,
                               "metrics/mean_acc": acc,
                               "loss/lovasz_loss": tf.metrics.mean(loss)}

                s_op = []

                with tf.variable_scope(mode):
                    if self.data_format == "NCHW":
                        s_op.extend([tf.summary.image(f"{mode}_image",
                                                      tf.transpose(input[..., :1],
                                                                   perm=[0, 2, 3, 1]),
                                                      max_outputs=1),
                                     tf.summary.image(f"{mode}_label",
                                                      tf.transpose(labels,
                                                                   perm=[0, 2, 3, 1]),
                                                      max_outputs=1),
                                     tf.summary.image(f"{mode}_prob",
                                                      tf.transpose(output,
                                                                   perm=[0, 2, 3, 1]),
                                                      max_outputs=1),
                                     tf.summary.image(f"{mode}_prediction",
                                                      tf.transpose(predicted,
                                                                   perm=[0, 2, 3, 1]),
                                                      max_outputs=1)])
                    else:
                        s_op.extend([tf.summary.image(f"{mode}_image",
                                                      input[..., :1],
                                                      max_outputs=1),
                                     tf.summary.image(f"{mode}_label",
                                                      labels,
                                                      max_outputs=1),
                                     tf.summary.image(f"{mode}_prob",
                                                      output,
                                                      max_outputs=1),
                                     tf.summary.image(f"{mode}_prediction",
                                                      predicted,
                                                      max_outputs=1)])

                if mode == tf.estimator.ModeKeys.TRAIN:

                    with tf.variable_scope("metrics"):
                        s_op.append(tf.summary.scalar('mean_acc', acc[1]))
                        s_op.append(tf.summary.scalar('mean_iou', iou[1]))

                    with tf.variable_scope("loss"):
                        s_op.append(tf.summary.scalar('lovasz_loss', loss))

                    lr = tf.train.exponential_decay(self.lr, tf.train.get_global_step(),
                                                    10000, 0.5, staircase=False,
                                                    name='learning_rate')
                    optimizer = tf.train.AdamOptimizer(lr)
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

                    # write out summary metrics and images to train
                    training_hook.append(tf.train.SummarySaverHook(
                        save_steps=20,
                        output_dir=f"{self.model_dir}/fold{fold}/train",
                        summary_op=tf.summary.merge(s_op)))

                else:

                    # write out summary images to eval
                    eval_hook.append(tf.train.SummarySaverHook(
                        save_steps=1,
                        output_dir=f"{self.model_dir}/fold{fold}/eval",
                        summary_op=tf.summary.merge(s_op)))

                    # save checkpoint if eval metric improves

                    train_op = None

            else:
                raise ValueError(f"Unknown mode {mode}")

            estimator = tf.estimator.EstimatorSpec(
                mode,
                predictions={"probabilities": output,
                             "mask": predicted},
                loss=loss,
                train_op=train_op,
                eval_metric_ops=evalmetrics,
                training_hooks=training_hook,
                evaluation_hooks=eval_hook
            )

            return estimator

        return _model_fn

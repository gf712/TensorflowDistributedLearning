import tensorflow as tf
from utils import get_available_gpus, metric_comparisson
from core.resnet import resnet_model
from core.losses import lovasz_loss
from core.metric import mIOU, mean_accuracy
from preprocessing.preprocessing import _prepare_directory, read_and_preprocess, single_transformation_from_jpeg, \
    single_transformation_from_matrix, create_symlinks
from sklearn.model_selection import StratifiedKFold
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


class Model:

    def __init__(self,
                 model_dir,
                 data_directory,
                 data_format="NHWC",
                 lr=0.001,
                 n_gpus=2,
                 n_fold=5,
                 seed=42,
                 save_best=5,
                 **kwargs):
        """
        High level class to perform multi GPU training with tf.contrib.distribute.MirroredStrategy
        and tf.Estimator.
        The base models are built using tf.slim.
        All the data is processed using tf.data.Dataset and tf.image.
        The preprocessing currently runs on the CPU (optimal?)
        Additionally built ResNet with NCHW format support for (potentially) faster GPU and MKL optimised CPU operations
        Currently NCHW format support is experimental and the speed up is minor (about 10%)


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

        if "weight_decay" in kwargs:
            self.weight_decay = kwargs["weight_decay"]
        else:
            self.weight_decay = WEIGHT_DECAY

        if "batch_norm_decay" in kwargs:
            self.batch_norm_decay = kwargs["weight_decay"]
        else:
            self.batch_norm_decay = BATCH_NORM_DECAY

        if "batch_norm_epsilon" in kwargs:
            self.batch_norm_epsilon = kwargs["batch_norm_epsilon"]
        else:
            self.batch_norm_epsilon = BATCH_NORM_EPSILON

        if "batch_norm_scale" in kwargs:
            self.batch_norm_scale = kwargs["batch_norm_scale"]
        else:
            self.batch_norm_scale = BATCH_NORM_SCALE

        if "output_stride" in kwargs:
            self.output_stride = kwargs["output_stride"]
        else:
            self.output_stride = OUTPUT_STRIDE

        if "base_depth" in kwargs:
            self.base_depth = kwargs["base_depth"]
        else:
            self.base_depth = BASE_DEPTH

        if "input_shape" in kwargs:
            self.input_shape = kwargs["input_shape"]
        else:
            self.input_shape = INPUT_SHAPE

        if "n_blocks" in kwargs:
            self.n_blocks = kwargs["n_blocks"]
        else:
            self.n_blocks = (3, 4, 6)

        if "block_type" in kwargs:
            self.block_type = kwargs["block_type"]
        else:
            self.block_type = "bottleneck"

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

        self.save_best = save_best

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
                        'threshold': 0.5,  # probability threshold for positive classification
                        'block_type': "basic_block"
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

            # checkpoint best model and keep up to self.save_best models
            if self.save_best > 0:
                def serving_input_receiver_fn():
                    inputs = {
                        "image": tf.placeholder(tf.float32, [None, 101, 101, 2]),
                    }
                    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

                exporter = tf.estimator.BestExporter(name="best_exporter",
                                                     serving_input_receiver_fn=serving_input_receiver_fn,
                                                     exports_to_keep=self.save_best,
                                                     compare_fn=functools.partial(metric_comparisson,
                                                                                  key="metrics/mean_iou",
                                                                                  greater_is_better=True)
                                                     )
            else:
                exporter = None

            # for inference increasing the batch size to double should not cause any issues
            eval_spec = tf.estimator.EvalSpec(
                input_fn=self._make_input_fn(mode=tf.estimator.ModeKeys.EVAL,
                                             fold=i,
                                             batch_size=batch_size * 2,
                                             augment=False,
                                             shuffle=False),
                steps=len(test_index) // (batch_size * 2),
                throttle_secs=300,
                start_delay_secs=0,
                exporters=exporter
            )

            tf.estimator.train_and_evaluate(
                model,
                train_spec,
                eval_spec
            )

            tf.logging.info(f'Finished training fold {i}.')

            i += 1

    # TODO: finish writing this method
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

            dataset = dataset.map(map_func=functools.partial(single_transformation_from_jpeg,
                                                             transformation=tti))
            tf.logging.info(dataset)

            # dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(self.n_gpus * 2)  # prefetch twice as many batches as there are GPUs

            return dataset

        return test_input_fn

    def _make_input_fn(self, mode, fold, batch_size, augment, shuffle):

        def input_fn():
            # A vector of filenames.
            filenames = tf.constant(
                tf.gfile.Glob(f'{self.model_dir}/{mode}/images/fold{fold}/*.png'))

            # `labels[i]` is the label for the image in `filenames[i].
            labels = tf.constant(
                tf.gfile.Glob(f'{self.model_dir}/{mode}/masks/fold{fold}/*.png'))

            dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

            # based on:
            # https://medium.com/tensorflow/multi-gpu-training-with-estimators-tf-keras-and-tf-data-ba584c3134db
            if shuffle:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
                    buffer_size=batch_size * 10,
                    count=None)
                )
            else:
                dataset = dataset.repeat()

            # for some reason this is slower than the line of code after calling read_and_preprocess
            # dataset = dataset.apply(
            #     tf.contrib.data.map_and_batch(map_func=functools.partial(read_and_preprocess, augment=augment),
            #                                   batch_size=batch_size,
            #                                   num_parallel_calls=os.cpu_count()
            #                                   ))

            dataset = dataset.map(functools.partial(read_and_preprocess,
                                                    augment=augment,
                                                    crop_probability=0)).batch(batch_size)

            # prefetch twice as many batches as there are GPUs -> doesn't always help and might need adjustment
            dataset = dataset.prefetch(self.n_gpus * 2)

            return dataset

        return input_fn

    def build_model_fn_optimizer(self):

        def _model_fn(features, labels, mode, params):

            if "fold" in params:
                fold = params["fold"]
            # treshold is used to determine when a prediction is negative/positive (0/1)
            if "threshold" in params:
                threshold = params["threshold"]
            else:
                threshold = 0.5

            # switch between training and inference phase for batch norm
            if mode == tf.estimator.ModeKeys.TRAIN:
                is_training = True
            else:
                is_training = False

            if self.data_format == "NCHW":
                input = tf.transpose(features["images"], [0, 3, 1, 2], name="image_input")
                if mode != tf.estimator.ModeKeys.PREDICT:
                    labels = tf.transpose(labels, [0, 3, 1, 2], name="label_input")
            else:
                input = tf.identity(features["images"], name="image_input")
                if mode != tf.estimator.ModeKeys.PREDICT:
                    labels = tf.identity(labels, name="label_input")

            with tf.variable_scope(self.model_name):

                # Model definition
                preactivation_output = resnet_model(
                    input=input,
                    model_name=self.model_name,
                    weight_decay=self.weight_decay,
                    batch_norm_decay=self.batch_norm_decay,
                    batch_norm_epsilon=self.batch_norm_epsilon,
                    batch_norm_scale=self.batch_norm_scale,
                    data_format=self.data_format,
                    is_training=is_training,
                    output_stride=self.output_stride,
                    base_depth=self.base_depth,
                    input_shape=self.input_shape,
                    n_blocks=self.n_blocks,
                    block_type=self.block_type
                )
                output = tf.nn.sigmoid(preactivation_output, name='sigmoid_output')
                predicted = tf.to_float(tf.greater(output, threshold))

            # maintenance code
            # hooks to keep track of training and evaluation
            eval_hook = []
            training_hook = []

            if mode == tf.estimator.ModeKeys.PREDICT:
                loss = None
                train_op = None
                evalmetrics = None

                # reverse test time augmentation -> this works for all flipping of images along a given axis
                # transpose, horizontal and vertical flip (and no augmentation)
                output = single_transformation_from_matrix(output, params["transformation"])["images"]
                predicted = single_transformation_from_matrix(predicted, params["transformation"])["images"]

            elif mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.TRAIN):

                with tf.device("/device:CPU:0"):
                    loss = lovasz_loss(labels,
                                       preactivation_output,
                                       data_format=self.data_format)

                # metrics to keep track of
                iou = mIOU(labels, predicted, name="mean_iou_metric")
                acc = mean_accuracy(labels, predicted, name="mean_acc_metric")

                # add loss to keep track on the same plot as training data on tensorboard
                evalmetrics = {"metrics/mean_iou": iou,
                               "metrics/mean_acc": acc,
                               "loss/lovasz_loss": tf.metrics.mean(loss)}

                # list with tf.summary ops
                s_op = []

                with tf.variable_scope(mode):
                    # write out images (convert to NHWC format, only format supported by tf.image)
                    if self.data_format == "NCHW":
                        s_op.extend([tf.summary.image(f"{mode}_image",
                                                      tf.transpose(input,
                                                                   perm=[0, 2, 3, 1])[..., :1],
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

                    # get number of parameters in model
                    self.n_params = sum(functools.reduce(lambda x, y: x * y, x.get_shape().as_list()) for x in
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model_name))

                    # add metric ops to add to train hooks -> this allows eval and train set metrics to be in
                    # the same plot in tensorboard
                    with tf.variable_scope("metrics"):
                        s_op.append(tf.summary.scalar('mean_acc', acc[1]))
                        s_op.append(tf.summary.scalar('mean_iou', iou[1]))

                    with tf.variable_scope("loss"):
                        s_op.append(tf.summary.scalar('lovasz_loss', loss))

                    # define learning rate schedule
                    lr = tf.train.exponential_decay(self.lr, tf.train.get_global_step(),
                                                    10000, 0.5, staircase=False,
                                                    name='learning_rate')

                    # define optimiser
                    optimizer = tf.contrib.optimizer_v2.AdamOptimizer(lr)

                    # add batch norm moving average and variance to update ops
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

                    # no training op required for validation set
                    train_op = None

            else:
                # only available modes are TRAIN, EVAL or PREDICT
                raise ValueError(f"Unknown mode {mode}")

            # instantiate estimator spec with all the things defined above
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

        # return function with signature f(features, labels, mode, params): -> tf.estimator.EstimatorSpec
        return _model_fn

    @property
    def params(self):
        try:
            return self.n_params
        except:
            raise ValueError("No model has been defined at this point! Call train method first.")

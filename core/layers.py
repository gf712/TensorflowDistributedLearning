import tensorflow as tf


slim = tf.contrib.slim

@slim.add_arg_scope
def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):

    """Splits a separable conv2d into depthwise and pointwise conv2d.
    This operation differs from `tf.layers.separable_conv2d` as this operation
    applies activation function between depthwise and pointwise conv2d.
    Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.
    Returns:
    Computed features after split separable conv2d.
    """
    outputs = slim.separable_conv2d(inputs,
                                    None,
                                    kernel_size=kernel_size,
                                    depth_multiplier=1,
                                    rate=rate,
                                    weights_initializer=tf.truncated_normal_initializer(
                                        stddev=depthwise_weights_initializer_stddev),
                                    weights_regularizer=None,
                                    scope=scope + '_depthwise')
    return slim.conv2d(outputs,
                       filters,
                       1,
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=pointwise_weights_initializer_stddev),
                       weights_regularizer=slim.l2_regularizer(weight_decay),
                       scope=scope + '_pointwise')


@slim.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs

@slim.add_arg_scope
def _upsample(inputs, out_shape, data_format='NCHW'):
    # we need to pad with one pixel, so we set kernel_size = 3
    inputs = _fixed_padding(inputs, 3, mode='SYMMETRIC', data_format=data_format)

    # tf.image.resize_bilinear accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    height = out_shape[1]
    width = out_shape[0]

    # we padded with 1 pixel from each side and upsample by factor of 2, so new dimensions will be
    # greater by 4 pixels after interpolation
    new_height = height + 4
    new_width = width + 4

    inputs = tf.image.resize_bilinear(inputs, (new_height, new_width))

    # trim back to desired size
    inputs = inputs[:, 2:-2, 2:-2, :]

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs

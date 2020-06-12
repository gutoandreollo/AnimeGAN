import tensorflow as tf
import tensorflow.contrib as tf_contrib

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)


weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.compat.v1.variable_scope(scope):
        if (kernel - stride) % 2 == 0 :
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else :
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.compat.v1.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.compat.v1.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    with tf.compat.v1.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], tf.shape(x)[1]*stride, tf.shape(x)[2]*stride, channels]
        if sn :
            w = tf.compat.v1.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

            if use_bias :
                bias = tf.compat.v1.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias)

        return x


def conv2d(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias = None):
    if kernel_size == 3:
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    return tf.contrib.layers.conv2d(
        inputs,
        num_outputs=filters,
        kernel_size=kernel_size,
        stride=strides,
        biases_initializer= Use_bias,
        normalizer_fn=None,
        activation_fn=None,
        padding=padding)


def conv2d_norm_lrelu(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias = None):
    x = conv2d(inputs, filters, kernel_size, strides, padding=padding, Use_bias = Use_bias)
    x = instance_norm(x,scope=None)
    return lrelu(x)


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=[1, 1, 1, 1],
               padding='VALID', stddev=0.02, name='dwise_conv', bias=False):
    input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    with tf.compat.v1.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        w = tf.compat.v1.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],regularizer=None,initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=name, data_format=None)
        if bias:
            biases = tf.compat.v1.get_variable('bias', [in_channel * channel_multiplier],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def separable_conv2d(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias = None):
    if kernel_size==3 and strides==1:
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    if strides == 2:
        inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="REFLECT")
    return tf.contrib.layers.separable_conv2d(
        inputs,
        num_outputs=filters,
        kernel_size=kernel_size,
        depth_multiplier=1,
        stride=strides,
        biases_initializer=Use_bias,
        normalizer_fn=tf.contrib.layers.instance_norm,
        activation_fn=lrelu,
        padding=padding)


def conv2d_transpose_lrelu(inputs, filters, kernel_size=2, strides=2, padding='SAME', Use_bias = None):
    return tf.contrib.layers.conv2d_transpose(inputs,
                                              num_outputs=filters,
                                              kernel_size=kernel_size,
                                              stride=strides,
                                              biases_initializer=Use_bias,
                                              normalizer_fn=tf.contrib.layers.instance_norm,
                                              activation_fn=lrelu,
                                              padding=padding)


def vgg_conv4_4_no_activation(vgg, img):
    vgg.build(img)
    return vgg.conv4_4_no_activation


##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    with tf.compat.v1.variable_scope(scope):
        with tf.compat.v1.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.compat.v1.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init


def invresblock(input, expansion_ratio, output_dim, stride, name, reuse=False, bias=None):
    with  tf.compat.v1.variable_scope(name, reuse=reuse):
        # pw
        bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
        net = conv2d_norm_lrelu(input, bottleneck_dim, kernel_size=1, Use_bias=bias)

        # dw
        net = dwise_conv(net, name=name)
        net = instance_norm(net,scope='1')
        net = lrelu(net)

        # pw & linear
        net = conv2d(net, output_dim, kernel_size=1)
        net = instance_norm(net,scope='2')

        # element wise add, only for stride==1
        if (int(input.get_shape().as_list()[-1]) == output_dim) and stride == 1:
            net = input + net

        return net


def flatten(x) :
    return tf.layers.flatten(x)


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


def sigmoid(x) :
    return tf.sigmoid(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.compat.v1.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def L2_loss(x,y):
    size = tf.size(x)
    return tf.nn.l2_loss(x-y)* 2 / tf.to_float(size)


def Huber_loss(x,y):
    return tf.compat.v1.losses.huber_loss(x,y)


def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


##################################################################################
# Image manipulation
##################################################################################

def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb = (rgb + 1.0)/2.0
    # rgb2yuv_filter = tf.constant([[[[0.299, -0.169, 0.499],
    #                                 [0.587, -0.331, -0.418],
    #                                 [0.114, 0.499, -0.0813]]]])
    # rgb2yuv_bias = tf.constant([0., 0.5, 0.5])
    # temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    # temp = tf.nn.bias_add(temp, rgb2yuv_bias)
    # return temp
    return tf.image.rgb_to_yuv(rgb)


def unsample(inputs, filters, kernel_size=3):
    '''
        An alternative to transposed convolution where we first resize, then convolve.
        See http://distill.pub/2016/deconv-checkerboard/
        For some reason the shape needs to be statically known for gradient propagation
        through tf.image.resize_images, but we only know that for fixed image size, so we
        plumb through a "training" argument
        '''
    new_H, new_W = 2 * tf.shape(inputs)[1], 2 * tf.shape(inputs)[2]
    inputs = tf.image.resize(inputs, [new_H, new_W])

    return separable_conv2d(filters=filters, kernel_size=kernel_size, inputs=inputs)


def downsample(inputs, filters = 256, kernel_size=3):
    '''
        An alternative to transposed convolution where we first resize, then convolve.
        See http://distill.pub/2016/deconv-checkerboard/
        For some reason the shape needs to be statically known for gradient propagation
        through tf.image.resize_images, but we only know that for fixed image size, so we
        plumb through a "training" argument
        '''

    new_H, new_W =  tf.shape(inputs)[1] // 2, tf.shape(inputs)[2] // 2
    inputs = tf.image.resize(inputs, [new_H, new_W])

    return separable_conv2d(filters=filters, kernel_size=kernel_size, inputs=inputs)


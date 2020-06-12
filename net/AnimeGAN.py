from ops import *
from utils import *
import time
import numpy as np
import json

from tensorflow.contrib import slim
from net.dataset import load_dataset
from net.vgg19 import Vgg19
from benedict import benedict
from tqdm import tqdm

class AnimeGAN(object) :
    def __init__(self, sess, params):
        default_params = {
            "general": {
                "gan_type": "lsgan",  # [gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]
                "epochs": 131,
                "checkpoint_frequency": 5  # every n epochs
            },
            "input": {
                "img_size": [256, 256, 3],
                "batch_size": 4,
                "paths": {
                    "real_train": "datasets/real_train/",
                    "real_test": "datasets/real_eval",
                    "anime": "datasets/Hayao",
                    "checkpoints": "checkpoints/Hayao_WIP",
                    "logs": "logs/Hayao_WIP",
                },
            },
            "init": {
                "epochs": 1,
                "learning_rate": 1e-4,
            },
            "generator": {
                "learning_rate": 8e-5,
                "weights": {
                    "adversarial_loss": 300,
                    "content_loss": 1.5,
                    "style_loss": 3.0,
                    "color_loss": 10.0,
                },
            },
            "discriminator": {
                "learning_rate": 16e-5,
                "layer_number": 3,
                "base_channel_number": 64,  # base channel number per layer
                "training_frequency": 1,  # train the discriminator 1/n less times than the generator
                "use_spectral_norm": True,
                "weights": {
                    "adversarial_loss": 300,
                    "gradient_penalty": 10,
                },
            },
        }
        self.params = benedict(params.update(default_params))
        self.sess = sess
        self.start_epoch = 0

        self._build_model()
        self._load_checkpoint()

    def _build_model(self):
        # Placeholders

        self.real = tf.compat.v1.placeholder(tf.float32, [self.params.get("inputs.batch_size"), self.params.get("inputs.img_size")[0], self.params.get("inputs.img_size")[1], self.params.get("inputs.img_size")[2]], name='real')
        self.real_test = tf.compat.v1.placeholder(tf.float32, [1, None, None, self.params.get("inputs.img_size")[2]], name='real_test')
        self.anime = tf.compat.v1.placeholder(tf.float32, [self.params.get("inputs.batch_size"), self.params.get("inputs.img_size")[0], self.params.get("inputs.img_size")[1], self.params.get("inputs.img_size")[2]], name='anime')
        self.anime_grayscale = tf.compat.v1.placeholder(tf.float32, [self.params.get("inputs.batch_size"), self.params.get("inputs.img_size")[0], self.params.get("inputs.img_size")[1], self.params.get("inputs.img_size")[2]],name='anime_grayscale')
        self.anime_grayscale_smooth = tf.compat.v1.placeholder(tf.float32, [self.params.get("inputs.batch_size"), self.params.get("inputs.img_size")[0], self.params.get("inputs.img_size")[1], self.params.get("inputs.img_size")[2]], name='anime_grayscale_smooth')

        # Generator / Discriminators

        self.generator_real = self.generator_real(self.real)
        self.generator_test = self.generator_real(self.test_real, reuse=True)
        self.discriminator_generated = self.discriminator_generated(self.generator_real, reuse=True)
        self.discriminator_anime = self.discriminator_generated(self.anime)  # TODO: Reuse?
        self.discriminator_anime_grayscale = self.discriminator_generated(self.anime_gray, reuse=True)
        self.discriminator_anime_grayscale_smooth = self.discriminator_generated(self.anime_gray_smooth, reuse=True)

        # VGG Features & YUV

        vgg = Vgg19()
        real_features = vgg_conv4_4_no_activation(vgg, self.real)
        generator_real_features = vgg_conv4_4_no_activation(vgg, self.generator_real)
        anime_gray_features = vgg_conv4_4_no_activation(vgg, self.anime[:generator_real_features.shape[0]])

        real_yuv = rgb2yuv(self.real)
        generator_real_yuv = rgb2yuv(self.generator_real)

        # Losses

        content_loss = L1_loss(real_features, generator_real_features)  # L_con
        style_loss = L1_loss(gram(anime_gray_features), gram(generator_real_features))  # L_gra
        color_loss = \
            L1_loss(real_yuv[:,:,:,0], generator_real_yuv[:,:,:,0]) \
            + Huber_loss(real_yuv[:,:,:,1], generator_real_yuv[:,:,:,1]) \
            + Huber_loss(real_yuv[:,:,:,2], generator_real_yuv[:,:,:,2])  # L_col
        generator_adversarial_loss = self._build_generator_adversarial_loss()
        discriminator_adversarial_loss = self._build_discriminator_adversarial_loss()
        discriminator_gradient_penalty = self._build_gradient_penalty()
        init_loss = self.params.get("generator.weights.content_loss") * L1_loss(real_features, generator_real_features)

        generator_loss = \
            self.params.get("generator.weights.adversarial_loss") * generator_adversarial_loss \
            + self.params.get("generator.weights.content_loss") * content_loss \
            + self.params.get("generator.weights.style_loss") * style_loss \
            + self.params.get("generator.weights.color_loss") * color_loss

        discriminator_loss = \
            self.params.get("discrimnator.weights.adversarial_loss") * discriminator_adversarial_loss \
            + self.params.get("discrimnator.weights.gradient_penalty") * discriminator_gradient_penalty

        # Optimizers

        generator_variables = [var for var in tf.compat.v1.trainable_variables() if 'generator' in var.name]
        discriminator_variables = [var for var in tf.compat.v1.trainable_variables() if 'discriminator' in var.name]

        self.init_optimizer = tf.compat.v1.train.AdamOptimizer(self.params.get("init.learning_rate"), beta1=0.5, beta2=0.999).minimize(init_loss, var_list=generator_variables)
        self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(self.params.get("generator.learning_rate"), beta1=0.5, beta2=0.999).minimize(generator_loss, var_list=generator_variables)
        self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(self.params.get("discriminator.learning_rate"), beta1=0.5, beta2=0.999).minimize(discriminator_loss, var_list=discriminator_variables)

        # Summaries

        summary_init_loss = tf.compat.v1.summary.scalar("G_init", self.init_loss)

        summary_generator_loss = tf.compat.v1.summary.scalar("generator_loss", self.generator_loss)
        summary_content_loss = tf.compat.v1.summary.scalar("content_loss", content_loss)
        summary_style_loss = tf.compat.v1.summary.scalar("style_loss", style_loss)
        summary_color_loss = tf.compat.v1.summary.scalar("color_loss", color_loss)
        summary_generator_adversarial_loss = tf.compat.v1.summary.scalar("generator_adversarial_loss", generator_adversarial_loss)

        summary_discriminator_loss = tf.compat.v1.summary.scalar("discriminator_loss", self.discriminator_loss)
        summary_discriminator_adversarial_loss = tf.compat.v1.summary.scalar("discriminator_adversarial_loss", discriminator_adversarial_loss)
        summary_discriminator_gradient_penalty = tf.compat.v1.summary.scalar("discriminator_gradient_penalty", discriminator_gradient_penalty)

        # TODO: Can we remove the variables above?
        self.summaries = tf.summary.merge_all()

        # self.summary_init_loss_merge = tf.compat.v1.summary.merge([summary_init_loss])
        # self.summary_generator_loss_merge = tf.compat.v1.summary.merge([summary_generator_loss, summary_content_loss, summary_style_loss, summary_color_loss, summary_generator_adversarial_loss])
        # self.summary_discriminator_loss_merge = tf.compat.v1.summary.merge([summary_discriminator_loss, summary_discriminator_adversarial_loss, summary_discriminator_gradient_penalty])

        # Informations
        print("-- Variables")
        slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)
        print('-- FLOPs: {}'.format(tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation()).total_float_ops))

    def _build_generator(self, inputs, reuse=False, scope="generator"):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            with tf.compat.v1.variable_scope('b1'):
                x = conv2d_norm_lrelu(inputs, 64)
                x = conv2d_norm_lrelu(x, 64)
                x = separable_conv2d(x, 128, strides=2) + downsample(x, 128)

            with tf.compat.v1.variable_scope('b2'):
                x = conv2d_norm_lrelu(x, 128)
                x = separable_conv2d(x, 128)
                x = separable_conv2d(x, 256, strides=2) + downsample(x, 256)

            with tf.compat.v1.variable_scope('m'):
                x = conv2d_norm_lrelu(x, 256)
                x = invresblock(x, 2, 256, 1, 'r1')
                x = invresblock(x, 2, 256, 1, 'r2')
                x = invresblock(x, 2, 256, 1, 'r3')
                x = invresblock(x, 2, 256, 1, 'r4')
                x = invresblock(x, 2, 256, 1, 'r5')
                x = invresblock(x, 2, 256, 1, 'r6')
                x = invresblock(x, 2, 256, 1, 'r7')
                x = invresblock(x, 2, 256, 1, 'r8')
                x = conv2d_norm_lrelu(x, 256)

            with tf.compat.v1.variable_scope('u2'):
                x = unsample(x, 128)
                x = separable_conv2d(x, 128)
                x = conv2d_norm_lrelu(x, 128)

            with tf.compat.v1.variable_scope('u1'):
                x = unsample(x, 128)    # The number of the filters in this layer is 128 while it is 64 in the graph of the paper. Please refer to the code.
                x = conv2d_norm_lrelu(x, 64)
                x = conv2d_norm_lrelu(x, 64)

            return tf.tanh(conv2d(x, filters=3, kernel_size=1, strides=1))

    def _build_discriminator(self, inputs, reuse=False, scope="discriminator"):
        channel = self.params.get("discriminator.base_channel_number") // 2

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x = conv(inputs, channel, kernel=3, stride=1, pad=1, use_bias=False, sn=self.params.get("discriminator.use_spectral_norm"), scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.params.get("discriminator.layer_number")):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=False, sn=self.params.get("discriminator.use_spectral_norm"), scope='conv_s2_' + str(i))
                x = lrelu(x, 0.2)

                x = conv(x, channel * 4, kernel=3, stride=1, pad=1, use_bias=False, sn=self.params.get("discriminator.use_spectral_norm"), scope='conv_s1_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel * 2, kernel=3, stride=1, pad=1, use_bias=False, sn=self.params.get("discriminator.use_spectral_norm"), scope='last_conv')
            x = instance_norm(x, scope='last_ins_norm')
            x = lrelu(x, 0.2)
            x = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, sn=self.params.get("discriminator.use_spectral_norm"), scope='D_logit')

            return x

    def _build_generator_adversarial_loss(self):
        loss = 0

        if self.params.get("general.gan_type") == 'wgan-gp' or self.params.get("general.gan_type") == 'wgan-lp':
            loss = -tf.reduce_mean(self.discriminator_generated)
        if self.params.get("general.gan_type") == 'lsgan' :
            loss = tf.reduce_mean(tf.square(self.discriminator_generated - 1.0))
        if self.params.get("general.gan_type") == 'gan' or self.params.get("general.gan_type") == 'dragan':
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discriminator_generated), logits=self.discriminator_generated))
        if self.params.get("general.gan_type") == 'hinge':
            loss = -tf.reduce_mean(self.discriminator_generated)

        return loss

    def _build_discriminator_adversarial_loss(self):
        discriminator_anime_loss = 0
        discriminator_anime_grayscale_loss = 0
        discriminator_generated_loss = 0
        discriminator_anime_grascale_smooth_loss = 0

        if self.params.get("general.gan_type") == 'wgan-gp' or self.params.get("general.gan_type") == 'wgan-lp':
            discriminator_anime_loss = -tf.reduce_mean(self.discriminator_anime)
            discriminator_anime_grayscale_loss = tf.reduce_mean(self.discriminator_anime_grayscale)
            discriminator_generated_loss = tf.reduce_mean(self.discriminator_generated)
            discriminator_anime_grascale_smooth_loss = tf.reduce_mean(self.discriminator_anime_grayscale_smooth)

        if self.params.get("general.gan_type") == 'lsgan' :
            discriminator_anime_loss = tf.reduce_mean(tf.square(self.discriminator_anime - 1.0))
            discriminator_anime_grayscale_loss = tf.reduce_mean(tf.square(self.discriminator_anime_grayscale))
            discriminator_generated_loss = tf.reduce_mean(tf.square(self.discriminator_generated))
            discriminator_anime_grascale_smooth_loss = tf.reduce_mean(tf.square(self.discriminator_anime_grayscale_smooth))

        if self.params.get("general.gan_type") == 'gan' or self.params.get("general.gan_type") == 'dragan' :
            discriminator_anime_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.discriminator_anime), logits=self.discriminator_anime))
            discriminator_anime_grayscale_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.discriminator_anime_grayscale), logits=self.discriminator_anime_grayscale))
            discriminator_generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.discriminator_generated), logits=self.discriminator_generated))
            discriminator_anime_grascale_smooth_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.discriminator_anime_grayscale_smooth), logits=self.discriminator_anime_grayscale_smooth))

        if self.params.get("general.gan_type") == 'hinge':
            discriminator_anime_loss = tf.reduce_mean(relu(1.0 - self.discriminator_anime))
            discriminator_anime_grayscale_loss = tf.reduce_mean(relu(1.0 + self.discriminator_anime_grayscale))
            discriminator_generated_loss = tf.reduce_mean(relu(1.0 + self.discriminator_generated))
            discriminator_anime_grascale_smooth_loss = tf.reduce_mean(relu(1.0 + self.discriminator_anime_grayscale_smooth))

        return discriminator_anime_loss + discriminator_generated_loss + discriminator_anime_grayscale_loss + 0.1 * discriminator_anime_grascale_smooth_loss

    def _build_gradient_penalty(self, scope="discriminator"):
        if not self.params.get("general.gan_type").__contains__('gp') and not self.params.get("general.gan_type").__contains__('lp') and not self.params.get("general.gan_type").__contains__('dragan'):
            return 0

        if self.params.get("general.gan_type").__contains__('dragan') :
            eps = tf.random_uniform(shape=tf.shape(self.anime), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(self.anime, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            generator_real = self.anime + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.params.get("inputs.batch_size"), 1, 1, 1], minval=0., maxval=1.)
        interpolated = self.anime + alpha * (generator_real - self.anime)

        grad = tf.gradients(self.discriminator(interpolated, reuse=True, scope=scope), interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        if self.params.get("general.gan_type").__contains__('lp'):
            return tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))
        elif self.params.get("general.gan_type").__contains__('gp') or self.params.get("general.gan_type") == 'dragan':
            return tf.reduce_mean(tf.square(grad_norm - 1.))
        else:
            return 0

    def _load_checkpoint(self):
        # Load checkpoint
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            tf.compat.v1.train.Saver().restore(self.sess, os.path.join(self.checkpoint_dir, checkpoint_name))
            self.start_epoch = int(checkpoint_name.split('-')[-1])
            print("Successfully loaded checkpoint {}".format(checkpoint_name))

    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Load datasets
        real_image_iterator, real_image_iterator_num = load_dataset(self.params.get("inputs.paths.real_train"), self.params.get("inputs.batch_size"))
        anime_image_iterator, anime_image_iterator_num = load_dataset(self.params.get("inputs.paths.anime"), self.params.get("inputs.batch_size"))
        dataset_num = max(real_image_iterator_num, anime_image_iterator_num)

        for epoch in tqdm(range(self.start_epoch, self.params.get("general.epochs"))):
            j = self.params.get("discriminator.training_frequency")
            for idx in tqdm(range(int(dataset_num / self.params.get("inputs.batch_size")))):
                start_time = time.time()

                # Build dataset for the step
                real_imgs, anime_imgs = self.sess.run([real_image_iterator, anime_image_iterator])
                train_feed_dict = {
                    self.real: real_imgs[0],
                    self.anime: anime_imgs[0],
                    self.anime_gray: anime_imgs[1],
                    self.anime_gray_smooth: anime_imgs[3]
                }

                # Run init phase
                if epoch < self.params.get("init.epochs"):
                    _, summaries = self.sess.run([self.init_optimizer, self.summaries], feed_dict=train_feed_dict)

                # Train discriminator
                if j == self.params.get("discriminator.training_frequency"):
                    _, summaries = self.sess.run([self.discriminator_optimizer, self.summaries], feed_dict=train_feed_dict)

                # Train generator
                _, summaries = self.sess.run([self.generator_optimizer, self.summaries], feed_dict=train_feed_dict)

                # Write summaries & print status
                tf.compat.v1.summary.FileWriter(logs_path, self.sess.graph).add_summary(summaries, epoch)
                print("Epoch: %3d Step: %5d  time: %f init_loss: %.8f" % (epoch, idx, time.time() - start_time))

                # Update discriminator's training frequency counter
                j = j - 1
                if j < 1:
                    j = self.params.get("discriminator.training_frequency")

            # Make a checkpoint of the epoch
            if (epoch + 1) >= self.params.get("init.epochs") and np.mod(epoch + 1, self.params.get("general.checkpoint_frequency")) == 0:
                if not os.path.exists(self.params.get("inputs.paths.checkpoints")):
                    os.makedirs(self.params.get("inputs.paths.checkpoints"))
                tf.compat.v1.train.Saver(max_to_keep=self.params.get("inputs.general.epochs")).save(self.sess, os.path.join(self.params.get("inputs.paths.checkpoints"), self.__class__.__name__ + '.model'), epoch)
                with open(os.path.join(self.params.get("inputs.paths.checkpoints"), self.__class__.__name__ + '.json'), "w") as f:
                    json.dump(f, params, indent=4)

            # TODO: Replace with Image Summary.
            # if epoch >= self.params.get("init.epochs") -1:
            #     """ Result Image """
            #     val_files = glob(test_imgs_path) # './dataset/{}/*.*'.format('val')
            #     save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
            #     check_folder(save_path)
            #     for i, sample_file in enumerate(val_files):
            #         print('val: '+ str(i) + sample_file)
            #         sample_image = np.asarray(load_test_data(sample_file, self.params.get("inputs.img_size")))
            #         test_real,test_generated = self.sess.run([self.test_real,self.test_generated],feed_dict = {self.test_real:sample_image} )
            #
            #         save_images(test_real, save_path+'{:03d}_a.png'.format(i))
            #         save_images(test_generated, save_path+'{:03d}_b.png'.format(i))

    def transform(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        return 0
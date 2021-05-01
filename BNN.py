from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from os import path
import warnings

# warnings.simplefilter(action="ignore")
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backends import backend_agg
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
# tf.logging.set_verbosity(tf.logging.ERROR)
# Dependency imports
import matplotlib
import tensorflow_probability as tfp
import random

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from skimage.color import gray2rgb, rgb2gray, label2rgb  # since the code wants color images
from sklearn.datasets import fetch_openml
import sklearn
import cv2
from scipy.ndimage import gaussian_filter

import os, sys

try:
	import lime
except:
	sys.path.append(os.path.join('..', '..'))  # add the current directory
	import lime

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import pickle

# %matplotlib inline
# import tensorflow_datasets as tfds


# initializing parameters
# learning_rate = 0.001   #initial learning rate
# max_step = 5000 #number of training steps to run
# batch_size = 50 #batch size
# viz_steps = 500 #frequency at which save visualizations.
# num_monte_carlo = 50 #Network draws to compute predictive probabilities.


sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.2, 0.3, 0.5, 1]
maxs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


# code credit: https://medium.com/python-experiments/bayesian-cnn-model-on-mnist-data-using-tensorflow-probability-compared-to-cnn-82d56a298f45
def train_bcnn(mnist_conv, learning_rate=0.001, max_step=3000, batch_size=50, load=False, load_name='', save=False,
               save_name='', model='orig', sigma=0.1, min_noise=0, max_noise=1):
    if load and load_name == '':
        print("missing load_name")
        exit()
    if save and save_name == '':
        print("missing save_name")
        exit()

    # defining the model
    images = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels = tf.compat.v1.placeholder(tf.float32, shape=[None, ])
    hold_prob = tf.compat.v1.placeholder(tf.float32)
    neural_net = tf.keras.Sequential([
        tfp.layers.Convolution2DReparameterization(32, kernel_size=5, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tfp.layers.Convolution2DReparameterization(64, kernel_size=5, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(hold_prob),
        tfp.layers.DenseFlipout(10)])
    logits = neural_net(images)
    # Compute the -ELBO as the loss, averaged over the batch size.
    labels_distribution = tfp.distributions.Categorical(logits=logits)
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / mnist_conv.train.num_examples
    elbo_loss = neg_log_likelihood + kl
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(elbo_loss)
    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)

    # training
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())

    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        if load:
            saver = tf.compat.v1.train.import_meta_graph(load_name + '.meta')
            saver.restore(sess, load_name)
        # Run the training loop.
        #define noise
        GaussianNoise = tf.keras.layers.GaussianNoise(sigma)
        UniformNoise = tf.keras.layers.Lambda(lambda x: x + random.uniform(min_noise, max_noise))
        for step in range(max_step + 1):
            images_b, labels_b = mnist_conv.train.next_batch(
                batch_size)
            #images_h, labels_h = mnist_conv.validation.next_batch(
            #    mnist_conv.validation.num_examples)
            if model == 'gaussian':
                g_noise = np.random.normal(0, sigma, (batch_size, 28, 28, 1))
                g_noise = g_noise.reshape(batch_size, 28, 28, 1)
                images_b = images_b+ g_noise
                #images_h = sess.run(GaussianNoise(images_h))
            elif model == 'uniform':
                u_noise = np.random.uniform(low=min_noise, high=max_noise, size=(batch_size, 28, 28, 1))
                u_noise = u_noise.reshape(batch_size, 28, 28, 1)
                images_b = images_b + u_noise
                #images_h = sess.run(UniformNoise(images_h))
            else:
                pass


            _ = sess.run([train_op, accuracy_update_op], feed_dict={
                images: images_b, labels: labels_b, hold_prob: 0.5})

            if (step == 0) | (step % 500 == 0):

                loss_value, accuracy_value = sess.run([elbo_loss, accuracy], feed_dict={images: images_b,
                                                                                        labels: labels_b,
                                                                                        hold_prob: 0.5})

                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(step, loss_value, accuracy_value))
        # neural_net.save("train_orig")
        if save:
            save_path = saver.save(sess, save_name)
            print("Model saved in file: %s" % save_path)
        # print(images_b[0])
        # print(neural_net.predict(images_b))
        pred_results = sess.run(logits, feed_dict={images: images_b, hold_prob: 0.5})


def load_and_explain(load_name, learning_rate=0.001, model='orig', sigma=0.1, minimum=0, maximum=1):
	# defining the model
	images = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
	labels = tf.compat.v1.placeholder(tf.float32, shape=[None, ])
	hold_prob = tf.compat.v1.placeholder(tf.float32)
	# define the model
	neural_net = tf.keras.Sequential([
		tfp.layers.Convolution2DReparameterization(32, kernel_size=5, padding="SAME", activation=tf.nn.relu),
		tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
		tfp.layers.Convolution2DReparameterization(64, kernel_size=5, padding="SAME", activation=tf.nn.relu),
		tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
		tf.keras.layers.Flatten(),
		tfp.layers.DenseFlipout(1024, activation=tf.nn.relu),
		tf.keras.layers.Dropout(hold_prob),
		tfp.layers.DenseFlipout(10)])
	logits = neural_net(images)
	# Compute the -ELBO as the loss, averaged over the batch size.
	labels_distribution = tfp.distributions.Categorical(logits=logits)
	neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
	kl = sum(neural_net.losses) / mnist_conv.train.num_examples
	elbo_loss = neg_log_likelihood + kl
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(elbo_loss)
	# Build metrics for evaluation. Predictions are formed from a single forward
	# pass of the probabilistic layers. They are cheap but noisy predictions.
	predictions = tf.argmax(logits, axis=1)
	accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)

	# training
	init_op = tf.group(tf.compat.v1.global_variables_initializer(),
					   tf.compat.v1.local_variables_initializer())

	saver = tf.compat.v1.train.Saver()
	with tf.compat.v1.Session() as sess:
		sess.run(init_op)
		saver = tf.compat.v1.train.import_meta_graph(load_name + '.meta')
		saver.restore(sess, load_name)

		mnist = fetch_openml('mnist_784', version=1, cache=True)
		a = mnist.data.reshape((-1, 28, 28, 1))[0]

		X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))], 0)
		y_vec = mnist.target.astype(np.uint8)

		fig, ax1 = plt.subplots(1, 1)
		ax1.imshow(X_vec[0], interpolation='none')
		ax1.set_title('Digit: {}'.format(y_vec[0]))

		explainer = lime_image.LimeImageExplainer(verbose=False)
		segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

		def predict_wrap(x):
			test_images = rgb2gray(x)
			if x.ndim == 3:
				test_images = test_images.reshape(1, 28, 28, 1)
			else:
				n, _, _, _ = x.shape
				test_images = test_images.reshape(n, 28, 28, 1)
			test_images = test_images.astype(np.float32)
			logit_results = sess.run(logits, feed_dict={images: test_images, hold_prob: 0.5})
			distribution_results = sess.run(tf.nn.softmax(logit_results))

			return distribution_results

		gauss = np.random.normal(0, 0.1, (28, 28, 1))
		gauss = gauss.reshape(28, 28, 1)
		noisy = X_vec[0] + gauss
		# print(X_vec[0].shape)
		# print(noisy.shape)
		# exit()
		#saved_img_files = ['good_orig_misdetection_gaussian_0.01_82.png', 'good_orig_misdetection_gaussian_0.01_933_9_7.png', 'orig_misdetection_gaussian_0.01_15_8_5.png', 'orig_misdetection_gaussian_0.01_37_7_9.png', 'orig_misdetection_gaussian_0.01_133_9_5.png', 'orig_misdetection_gaussian_0.01_168_2_7.png', 'orig_misdetection_gaussian_0.01_254_9_7.png', 'orig_misdetection_gaussian_0.01_323_1_8.png', 'orig_misdetection_gaussian_0.01_351_4_9.png', 'orig_misdetection_gaussian_0.01_362_6_5.png', 'orig_misdetection_gaussian_0.01_405_8_5.png', 'orig_misdetection_gaussian_0.01_764_8_9.png', 'orig_misdetection_gaussian_0.01_789_3_5.png', 'orig_misdetection_gaussian_0.01_832_5_3.png', 'orig_misdetection_gaussian_0.01_835_6_5.png', 'orig_misdetection_gaussian_0.01_881_4_9.png', 'orig_misdetection_gaussian_0.01_901_9_7.png', 'orig_misdetection_gaussian_0.01_933_4_9.png', 'orig_misdetection_gaussian_0.01_983_0_1.png']
		#saved_img_files = ['orig_misdetection_gaussian_0.01_77_3_5.png', 'orig_misdetection_gaussian_0.01_87_8_0.png', 'orig_misdetection_gaussian_0.01_102_2_0.png', 'orig_misdetection_gaussian_0.01_137_3_1.png', 'orig_misdetection_gaussian_0.01_161_1_7.png', 'orig_misdetection_gaussian_0.01_172_9_0.png', 'orig_misdetection_gaussian_0.01_186_5_6.png', 'orig_misdetection_gaussian_0.01_318_9_4.png', 'orig_misdetection_gaussian_0.01_322_3_1.png', 'orig_misdetection_gaussian_0.01_323_9_0.png', 'orig_misdetection_gaussian_0.01_343_8_2.png', 'orig_misdetection_gaussian_0.01_411_9_0.png', 'orig_misdetection_gaussian_0.01_431_1_7.png', 'orig_misdetection_gaussian_0.01_435_6_8.png', 'orig_misdetection_gaussian_0.01_454_3_0.png', 'orig_misdetection_gaussian_0.01_483_2_1.png', 'orig_misdetection_gaussian_0.01_522_6_1.png', 'orig_misdetection_gaussian_0.01_567_8_9.png', 'orig_misdetection_gaussian_0.01_569_4_5.png', 'orig_misdetection_gaussian_0.01_634_9_0.png', 'orig_misdetection_gaussian_0.01_865_2_1.png', 'orig_misdetection_gaussian_0.01_876_3_0.png', 'orig_misdetection_gaussian_0.01_900_2_7.png', 'orig_misdetection_gaussian_0.01_982_6_0.png', 'orig_misdetection_gaussian_0.01_999_3_5.png']

		#for img_file in saved_img_files:
		#	x = cv2.imread(img_file)
		
		#	print(img_file, np.argmax(predict_wrap(x)))
		#exit()
		#good_example_1 = 'good_orig_misdetection_gaussian_0.01_82.png'
		#good_example_9 = 'good_orig_misdetection_gaussian_0.01_933_9_7.png'
		#good_example_6 = 'orig_misdetection_gaussian_0.01_362_6_5.png'
		#image_to_explain = cv2.imread(good_example_6)
		#image_to_explain_label = 6
		orig_322 = 'orig_322.png'
		gaussian_322 = 'orig_misdetection_gaussian_0.01_322_3_1.png'
		uniform_322 = 'uniform_0.01_322_3_1.png'
		#noise = np.random.uniform(low=0, high=0.01, size=(28, 28, 1))
		#noise = noise.reshape(28, 28, 1)
		image_to_explain_label = 3
		#cv2.imwrite('uniform_0.01_322_3_1.png', cv2.imread(orig_322) + noise)
		image_to_explain = cv2.imread(uniform_322)
		##print(image_to_explain.shape)
		#exit()
		#exit()
		explanation = explainer.explain_instance(image_to_explain,
												 classifier_fn=predict_wrap,
												 top_labels=10, hide_color=0, num_samples=10000,
												 segmentation_fn=segmenter)
		#pickle.dump()
		# exit()
		#pickle.dump(explanation, open("uniform0.01_test_on_uniform.pkl", 'wb'))
		#exit()

		# explanation = pickle.load(open("explanation.pkl", 'rb'))
		temp, mask = explanation.get_image_and_mask(image_to_explain_label, positive_only=True, num_features=10, hide_rest=False,
													min_weight=0.01)
		# print(temp, mask)
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
		#ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
		# print(label2rgb(mask,temp, bg_label = 0))
		#ax1.set_title('Positive Regions for {}'.format(image_to_explain_label))
		temp, mask = explanation.get_image_and_mask(image_to_explain_label, positive_only=False, num_features=10, hide_rest=False,
													min_weight=0.01)
		ax1.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
		ax1.set_title('Positive/Negative Regions for {}'.format(image_to_explain_label))
		temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False,
													min_weight=0.01)
		ax2.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
		ax2.set_title('Positive/Negative Regions for {}'.format(1))
		plt.savefig('orig_test_on_uniform0.01.png')
		# exit()
		# now show them for each class
		fig, m_axs = plt.subplots(2, 5, figsize=(12, 6))
		for i, c_ax in enumerate(m_axs.flatten()):
			temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=False,
														min_weight=0.01)
			c_ax.imshow(label2rgb(mask, image_to_explain, bg_label=0), interpolation='nearest')
			c_ax.set_title('Positive for {}\nActual {}'.format(i, image_to_explain_label))
			c_ax.axis('off')
		plt.savefig('all_class_orig_test_on_uniform0.01.png')
		return 0


def find_accuracy(load_name, validation_set, learning_rate=0.001, minimum=0):
    # print(validation_set.shape)
    # defining the model
    tf.reset_default_graph()
    with tf.Session() as sess:  # Create new session
        sess.run(tf.global_variables_initializer())
    images = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels = tf.compat.v1.placeholder(tf.float32, shape=[None, ])
    hold_prob = tf.compat.v1.placeholder(tf.float32)
    neural_net = tf.keras.Sequential([
        tfp.layers.Convolution2DReparameterization(32, kernel_size=5, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tfp.layers.Convolution2DReparameterization(64, kernel_size=5, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(hold_prob),
        tfp.layers.DenseFlipout(10)])
    logits = neural_net(images)
    # Compute the -ELBO as the loss, averaged over the batch size.
    labels_distribution = tfp.distributions.Categorical(logits=logits)
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / mnist_conv.train.num_examples
    elbo_loss = neg_log_likelihood + kl
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(elbo_loss)
    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)

    # training
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())
    validation_images = validation_set[0]
    validation_images = np.asarray(validation_images, dtype=np.float32)
    validation_labels = validation_set[1]

    with tf.compat.v1.Session() as sess:
        with open("{}_test".format(load_name), 'w+') as f:
            sess.run(init_op)
            saver = tf.compat.v1.train.import_meta_graph(load_name + '.meta')
            saver.restore(sess, load_name)

            # orig accuracy
            loss, predictions_orig = sess.run([elbo_loss, tf.argmax(logits, axis=1)],
                                               feed_dict={images: validation_images, labels: validation_labels,
                                                          hold_prob: 0.5})
            test_accuracy = sklearn.metrics.accuracy_score(validation_labels, predictions_orig)
            f.write("Accuracy and loss origin: {} {}\n".format(str(test_accuracy), str(loss)))

            '''
            # gaussian blur
            for blur_simga in [0.5, 0.8, 1.2, 1.5, 1.8, 2.1]:
                noisy_images = np.array([gaussian_filter(validation_image, blur_simga) for validation_image in validation_images])
                loss, predictions_blur = sess.run([elbo_loss, tf.argmax(logits, axis=1)],
                                                     feed_dict={images: noisy_images, labels: validation_labels,
                                                                hold_prob: 0.5})
                test_accuracy = sklearn.metrics.accuracy_score(validation_labels, predictions_blur)
                f.write(
                    "Accuracy and loss uniform with blur sigma {}: {} {}\n".format(str(blur_simga), str(test_accuracy), str(loss)))
            '''


            # gaussian accuracy
            for t_sigma in sigmas:
                GaussianNoise = tf.keras.layers.GaussianNoise(t_sigma)
                noisy_images = sess.run(GaussianNoise(validation_images))
                loss, predictions_gauss = sess.run([elbo_loss, tf.argmax(logits, axis=1)], feed_dict={images: noisy_images, labels: validation_labels,
                                                                                                      hold_prob: 0.5})
                #predictions_gauss = sess.run(tf.argmax(logits_gauss, axis=1))
                test_accuracy = sklearn.metrics.accuracy_score(validation_labels, predictions_gauss)
                f.write("Accuracy and loss gaussian with sigma {} : {} {}\n".format(str(t_sigma), str(test_accuracy),  str(loss)))

            # uniform accuracy
            for t_max in maxs:
                #UniformNoise = tf.keras.layers.Lambda(lambda x: x + random.uniform(minimum, t_max))
                noisy_images = validation_images + np.random.uniform(low=minimum, high=t_max, size=(len(validation_images), 28, 28, 1))
                loss, predictions_uniform = sess.run([elbo_loss, tf.argmax(logits, axis=1)],
                                                   feed_dict={images: noisy_images, labels: validation_labels,
                                                              hold_prob: 0.5})
                test_accuracy = sklearn.metrics.accuracy_score(validation_labels, predictions_uniform)
                f.write("Accuracy and loss uniform with max {}: {} {}\n".format(str(t_max), str(test_accuracy), str(loss)))


            #combination noise
            for t_sigma, t_max in zip(sigmas, maxs):
                GaussianNoise = tf.keras.layers.GaussianNoise(t_sigma)
                noisy_images = sess.run(GaussianNoise(validation_images))
                noisy_images = noisy_images + np.random.uniform(low=minimum, high=t_max,
                                                                     size=(len(validation_images), 28, 28, 1))
                loss, predictions_cobo = sess.run([elbo_loss, tf.argmax(logits, axis=1)],
                                                     feed_dict={images: noisy_images, labels: validation_labels,
                                                                hold_prob: 0.5})
                test_accuracy = sklearn.metrics.accuracy_score(validation_labels, predictions_cobo)
                f.write(
                    "Accuracy and loss uniform with combined sigma {} max {}: {} {}\n".format(str(t_sigma), str(t_max), str(test_accuracy), str(loss)))
    return 0


def load_and_test(load_name, learning_rate=0.001, model='orig', minimum=0, testing_num=1000,
                  model_name=""):
    # defining the model
    if model_name == "":
        model_name = model
    images = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels = tf.compat.v1.placeholder(tf.float32, shape=[None, ])
    hold_prob = tf.compat.v1.placeholder(tf.float32)
    neural_net = tf.keras.Sequential([
        tfp.layers.Convolution2DReparameterization(32, kernel_size=5, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tfp.layers.Convolution2DReparameterization(64, kernel_size=5, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(hold_prob),
        tfp.layers.DenseFlipout(10)])

    logits = neural_net(images)
    # Compute the -ELBO as the loss, averaged over the batch size.
    labels_distribution = tfp.distributions.Categorical(logits=logits)
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / mnist_conv.train.num_examples
    elbo_loss = neg_log_likelihood + kl
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(elbo_loss)
    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions)

    # training
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())

    # saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        saver = tf.compat.v1.train.import_meta_graph(load_name + '.meta')
        saver.restore(sess, load_name)

        mnist = fetch_openml('mnist_784', version=1, cache=True)

        X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))], 0)
        y_vec = mnist.target.astype(np.uint8)

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(X_vec[0], interpolation='none')
        ax1.set_title('Digit: {}'.format(y_vec[0]))

        def predict_wrap(x, noise):
            test_images = rgb2gray(x)

            if x.ndim == 3:
                test_images = test_images.reshape(1, 28, 28, 1)
            else:
                n, _, _, _ = x.shape
                test_images = test_images.reshape(n, 28, 28, 1)
            new_test_images = test_images + noise
            new_test_images = new_test_images.astype(np.float32)
            predictions = tf.argmax(logits, axis=1)
            logit_results = sess.run(predictions, feed_dict={images: new_test_images, hold_prob: 0.5})
            # distribution_results = sess.run(tf.nn.softmax(logit_results))

            return logit_results

        def extact_result(x, labels, noise):
            distribution_results = predict_wrap(x, noise)
            comparsion = lambda x, y: x == y
            predictions = comparsion(distribution_results, labels)
            predication_res = predictions[predictions == True]
            return float(len(predication_res) / len(x))

        sample_index = np.random.choice(len(X_vec), testing_num)
        test_x = X_vec[np.array(sample_index)] /255.0
        test_labels = y_vec[np.array(sample_index)]

        '''
                test for gaussian
                '''
        with open("{}_test".format(load_name), 'w+') as f:
            for t_sigma in sigmas:
                noise = np.random.normal(0, t_sigma, (testing_num, 28, 28, 1))
                noise = noise.reshape(testing_num, 28, 28, 1)
                test_accuracy = extact_result(test_x, test_labels, noise)
                f.write("Accuracy gaussian with sigma {}: {}\n".format(str(t_sigma), str(test_accuracy)))

	# training
	init_op = tf.group(tf.compat.v1.global_variables_initializer(),
					   tf.compat.v1.local_variables_initializer())

	# saver = tf.compat.v1.train.Saver()
	with tf.compat.v1.Session() as sess:
		sess.run(init_op)
		saver = tf.compat.v1.train.import_meta_graph(load_name + '.meta')
		saver.restore(sess, load_name)

		mnist = fetch_openml('mnist_784', version=1, cache=True)

		X_vec = np.stack([gray2rgb(iimg) for iimg in mnist.data.reshape((-1, 28, 28))], 0)
		y_vec = mnist.target.astype(np.uint8)

		fig, ax1 = plt.subplots(1, 1)
		ax1.imshow(X_vec[0], interpolation='none')
		ax1.set_title('Digit: {}'.format(y_vec[0]))

		def predict_wrap(x, noise):
			test_images = rgb2gray(x)

			if x.ndim == 3:
				test_images = test_images.reshape(1, 28, 28, 1)
			else:
				n, _, _, _ = x.shape
				test_images = test_images.reshape(n, 28, 28, 1)
			new_test_images = test_images + noise
			new_test_images = new_test_images.astype(np.float32)
			predictions = tf.argmax(logits, axis=1)
			logit_results = sess.run(predictions, feed_dict={images: new_test_images, hold_prob: 0.5})
			# distribution_results = sess.run(tf.nn.softmax(logit_results))

			return logit_results

		def extact_result(x, labels, noise):    
			distribution_results = predict_wrap(x, noise)
			comparsion = lambda x, y: x == y
			predictions = comparsion(distribution_results, labels)
			predication_res = predictions[predictions == True]
			wrong_pred_index = [i for i in range(len(x)) if not predictions[i]]
			print(noise)
			print(wrong_pred_index)
			list_file_names = []
			for i in wrong_pred_index:
				#print((x[i]+noise[i]).shape)
				#exit()
				cv2.imwrite('orig_misdetection_uniform_0.01_' + str(i) + '_'+str(labels[i]) + '_' + str(distribution_results[i]) + '.png', (x[i]+noise[i])*255)
				cv2.imwrite('orig_' + str(i) +'.png', (x[i])*255)

				list_file_names.append('orig_misdetection_uniform_0.01_' + str(i) + '_'+str(labels[i]) + '_' + str(distribution_results[i]) + '.png')
			print(list_file_names)
				#plt.imshow(x[i]+noise)
				#plt.savefig('orig_misdetection_' + str(i) + '.png')
			#cv2.imwrite('orig_362.png', (x[i])*255)

			return float(len(predication_res) / len(x))

		sample_index = np.random.choice(len(X_vec), testing_num)
		test_x = X_vec[np.array(sample_index)] /255.0
		test_labels = y_vec[np.array(sample_index)]

		'''
		test for gaussian 
		'''
		with open("{}_test".format(load_name), 'w+') as f:
			#for t_sigma in sigmas:
			#	noise = np.random.normal(0, t_sigma, (testing_num, 28, 28, 1))
			#	noise = noise.reshape(testing_num, 28, 28, 1)
			#	test_accuracy = extact_result(test_x, test_labels, noise)
			#	f.write("Accuracy gaussian with sigma {}: {}\n".format(str(t_sigma), str(test_accuracy)))
			for t_max in maxs:
				noise = np.random.uniform(low=minimum, high=t_max, size=(testing_num, 28, 28, 1))
				noise = noise.reshape(testing_num, 28, 28, 1)
				test_accuracy = extact_result(test_x, test_labels, noise)
				f.write("Accuracy uniform with max {}: {}\n".format(str(t_max), str(test_accuracy)))

			exit()
			noise = np.zeros(shape=(testing_num, 28, 28, 1))
			test_accuracy = extact_result(test_x, test_labels, noise)
			f.write("Accuracy origin: {}\n".format(str(test_accuracy)))


def train_orig():
	orig_model_save = "./saved_models/orig_model.ckpt"
	tf.reset_default_graph()
	with tf.Session() as sess:  # Create new session
		sess.run(tf.global_variables_initializer())
	if not path.exists(orig_model_save + ".meta"):
		train_bcnn(mnist_conv, save=True, save_name=orig_model_save)
	find_accuracy(orig_model_save, mnist_conv.validation.next_batch(mnist_conv.validation.num_examples))


# load_and_explain(orig_model_save)
# load_and_explain(orig_model_save, model='uniform', minimum=0, maximum=1)

def train_gaussian(noise_sigma):
	gaussian_model_1 = "./saved_models_gaussian_{}/gaussian_{}_model.ckpt".format(str(noise_sigma), str(noise_sigma))
	print("before running gaussian")
	tf.reset_default_graph()
	with tf.Session() as sess:  # Create new session
		sess.run(tf.global_variables_initializer())
	if not path.exists(gaussian_model_1 + ".meta"):
		train_bcnn(mnist_conv, save=True, save_name=gaussian_model_1, model='gaussian', sigma=noise_sigma)
	find_accuracy(gaussian_model_1, mnist_conv.validation.next_batch(mnist_conv.validation.num_examples))


# load_and_explain(gaussian_model_1, model='gaussian', sigma=noise_sigma)
# load_and_explain(gaussian_model_1, model='uniform', minimum=0, maximum=1)


def train_uniform(max):
	uniform_model_1 = "./saved_models_uniform_{}/uniform_{}_model.ckpt".format(str(max), str(max))
	print("before running uniform")
	tf.reset_default_graph()
	with tf.Session() as sess:  # Create new session
		sess.run(tf.global_variables_initializer())
	if not path.exists(uniform_model_1 + ".meta"):
		train_bcnn(mnist_conv, save=True, save_name=uniform_model_1, model='uniform', max_noise=max)
	find_accuracy(uniform_model_1, mnist_conv.validation.next_batch(mnist_conv.validation.num_examples))


# load_and_explain(uniform_model_1, model='uniform', minimum=0, maximum=max)
# load_and_explain(uniform_model_1, model='gaussian', sigma=0.1)
def display_results():
	pkl_files = ['orig_test_on_orig.pkl', 'orig_test_on_gaussian0.01.pkl', 'orig_test_on_uniform0.01.pkl',  'gaussian0.01_test_on_orig.pkl', 'gaussian0.01_test_on_gaussian.pkl', 'gaussian0.01_test_on_uniform.pkl', 'uniform0.01_test_on_orig.pkl', 'uniform0.01_test_on_gaussian.pkl', 'uniform0.01_test_on_uniform.pkl']
	orig_322 = 'orig_322.png'
	gaussian_322 = 'orig_misdetection_gaussian_0.01_322_3_1.png'
	uniform_322 = 'uniform_0.01_322_3_1.png'
	#noise = np.random.uniform(low=0, high=0.01, size=(28, 28, 1))
	#noise = noise.reshape(28, 28, 1)
	image_to_explain_label = 3
	#cv2.imwrite('uniform_0.01_322_3_1.png', cv2.imread(orig_322) + noise)
	image_to_explain = cv2.imread(uniform_322)
	##print(image_to_explain.shape)
	#exit()
	#exit()
	#explanation = explainer.explain_instance(image_to_explain,
	#										 classifier_fn=predict_wrap,
	#										 top_labels=10, hide_color=0, num_samples=10000,
	#										 segmentation_fn=segmenter)
	#pickle.dump()
	# exit()
	#pickle.dump(explanation, open("uniform0.01_test_on_uniform.pkl", 'wb'))
	#exit()

	# explanation = pickle.load(open("explanation.pkl", 'rb'))
	#3temp, mask = explanation.get_image_and_mask(image_to_explain_label, positive_only=True, num_features=10, hide_rest=False,
	#											min_weight=0.01)
	# print(temp, mask)
	#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
	#ax1.imshow(label2rgb(mask, temp, bg_label=0), interpolation='nearest')
	# print(label2rgb(mask,temp, bg_label = 0))
	#ax1.set_title('Positive Regions for {}'.format(image_to_explain_label))
	#temp, mask = explanation.get_image_and_mask(image_to_explain_label, positive_only=False, num_features=10, hide_rest=False,
	#											min_weight=0.01)
	#ax1.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
	#ax1.set_title('Positive/Negative Regions for {}'.format(image_to_explain_label))
	#temp, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False,
#												min_weight=0.01)
#	ax2.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
#	ax2.set_title('Positive/Negative Regions for {}'.format(1))
#	plt.savefig('orig_test_on_uniform0.01.png')
	# exit()
	# now show them for each class
	fig, m_axs = plt.subplots(3, 3, figsize=(24, 24))
	for i, c_ax in enumerate(m_axs.flatten()):
		explanation  = pickle.load(open(pkl_files[i], 'rb'))
		temp, mask = explanation.get_image_and_mask(image_to_explain_label, positive_only=True, num_features=10, hide_rest=False,
												min_weight=0.01)

		#temp, mask = explanation.get_image_and_mask(image_to_explain_label, positive_only=False, num_features=10, hide_rest=False,
		#										min_weight=0.01)

		#temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=False,
		#											min_weight=0.01)
		c_ax.imshow(label2rgb(3 - mask, temp, bg_label=0), interpolation='nearest')
		#c_ax.imshow(label2rgb(mask, image_to_explain, bg_label=0), interpolation='nearest')
		#c_ax.set_title('Positive for {}\nActual {}'.format(i, image_to_explain_label))
		c_ax.axis('off')
	plt.savefig('all_experiment_results.png')
	return 0

if __name__ == '__main__':
	display_results()
	exit()
	# loading dataset
	data_dir = "MNIST_data/"
	# mnist_onehot = input_data.read_data_sets(data_dir, one_hot=True)
	mnist_conv = input_data.read_data_sets(data_dir, reshape=False, one_hot=False)
	mnist_conv_onehot = input_data.read_data_sets(data_dir, reshape=False, one_hot=True)
	# print(mnist_conv.shape())
	# display an image
	# img_no = 485
	# one_image = mnist_conv_onehot.train.images[img_no].reshape(28,28)
	# print(one_image.shape())
	# exit()
	# plt.imshow(one_image, cmap='gist_gray')
	# print('Image label: {}'.format(np.argmax(mnist_conv_onehot.train.labels[img_no])))
	#load_and_explain(load_name, learning_rate=0.001, model='orig', sigma=0.1, minimum=0, maximum=1):
	#load_and_test('saved_models/orig_model.ckpt')
	#load_and_explain('/Users/caroline/Desktop/courses/2020-2021/412/project/Bayesian_NN_robustness/saved_models_gaussian_0.01/gaussian_0.01_model.ckpt')
	#load_and_explain('saved_models/orig_model.ckpt')
	load_and_explain('/Users/caroline/Desktop/courses/2020-2021/412/project/Bayesian_NN_robustness/saved_models_uniform_0.01/uniform_0.01_model.ckpt')
	exit()
	train_orig()

	for sigma in sigmas:
		train_gaussian(sigma)

	for max in maxs:
		train_uniform(max)
	exit()
	
	train_orig = False

	orig_model_save = "./saved_models/orig_model.ckpt"
	if train_orig:
		train_bcnn(mnist_conv, save=False, save_name=orig_model_save)
		
		#load_and_explain(orig_model_save)

	sigmas = [0.001, 0.1, 0.5, 0.9]
	train_gaussian = False
	if train_gaussian:
		for s in sigmas:
			gaussian_model_1 = "./saved_models/orig_gaussian_0.001_model.ckpt"
			print("before running gaussian")
			train_bcnn(mnist_conv, s, save=True, save_name=gaussian_model_1, model='gaussian')
	#load_and_explain(gaussian_model_1, model='gaussian', sigma=0.1)
	
	train_uniform = False
	maximum = [0.001, 0.1, 0.5, 0.9]

	if train_uniform:
		for m in maximum:
			uniform_model_1 = "./saved_models/uniform_" + str(m) +"_model.ckpt"
			print("before running uniform")
			train_bcnn(mnist_conv, min_noise=0, max_noise=m, save=True, save_name=uniform_model_1, model='uniform')
		#load_and_explain(uniform_model_1, model='uniform', minimum=0, maximum=0.01)
	#train_bcnn(mnist_conv, max_step=500, load=True, load_name=orig_model_save)
	gaussian_model_1 = "./saved_models/orig_gaussian_0.001_model.ckpt"
	uniform_model_1 = "./saved_models/uniform_0.1_model.ckpt"
	find_accuracy(uniform_model_1, mnist_conv.validation.next_batch(mnist_conv.validation.num_examples), maximum=0.1)


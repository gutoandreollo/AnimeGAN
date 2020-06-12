import os
import tensorflow as tf
import cv2
import numpy as np


# Returns a TF iterator of images given the specified paths, as well as the number of images available.
# Each iteration returns 4 normalized images: original, grayscale, smooth and smooth grayscale
def load_dataset(dataset_dir_path, batch_size):
    def list_images(img_dir_path):
        paths = []
        for path in os.listdir(img_dir_path):
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                continue
            paths.append(os.path.join(img_dir_path, path))
        return paths

    def smooth(original_img, grayscale_img):
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        gauss = cv2.getGaussianKernel(kernel_size, 0)
        gauss = gauss * gauss.transpose(1, 0)

        original_padded = np.pad(original_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        grayscale_edges = cv2.Canny(grayscale_img, 100, 200)
        grayscale_edges_dilated = cv2.dilate(grayscale_edges, kernel)

        smooth_img = np.copy(original_img)
        idx = np.where(grayscale_edges_dilated != 0)
        for i in range(np.sum(grayscale_edges_dilated != 0)):
            smooth_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(original_padded[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            smooth_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(original_padded[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            smooth_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(original_padded[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        return smooth_img

    def load_image(img_path):
        original_img = cv2.imread(img_path.decode()).astype(np.float32)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grayscale_img = np.asarray([grayscale_img, grayscale_img, grayscale_img])
        grayscale_img = np.transpose(grayscale_img, (1, 2, 0))

        smooth_img = smooth(original_img, grayscale_img)

        smooth_grayscale_img = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        smooth_grayscale_img = np.asarray([smooth_grayscale_img, grayscale_img, grayscale_img])
        smooth_grayscale_img = np.transpose(smooth_grayscale_img, (1, 2, 0))

        return original_img / 127.5 - 1.0, grayscale_img / 127.5 - 1.0, smooth_img / 127.5 - 1.0, smooth_grayscale_img / 127.5 - 1.0

    img_paths = list_images(dataset_dir_path)

    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=len(img_paths))
    dataset = dataset.map(lambda img_path: tf.py_func(load_image, [img_path], [tf.float32, tf.float32, tf.float32, tf.float32]), tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next(), len(img_paths)

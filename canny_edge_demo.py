# -*- coding: utf-8 -*-

"""
Different common functions for training the models.
Copyright (C) 2020, Daniel Kurniadi <daniel.thekurniadi@gmail.com>
This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import argparse

# core library
import math
import numpy as np  # store image in scalable matrix structure
from PIL import Image  # reading image from file

import matplotlib.pyplot as plt


def canny_edge_detect(image, kernel_size, kernel_sigma, lower_threshold, upper_threshold):
    use_degree = True

    # perform gaussian smoothing
    print("(canny)> Gaussian smoothing ...")
    smooth = gaussian_smoothing(image, kernel_size)

    # perform sobel edge detection
    print("(canny)> Sobel edge detection ...")
    gradient_magnitude, gradient_direction = sobel_edge_detection(smooth, use_degree)

    # perform non-max suppression
    print("(canny)> Non-Max Supp, Double thresholding, & Hysteresis edge ...")
    maxsup = non_max_suppression(gradient_magnitude, gradient_direction)
    edged_image = double_threshold(maxsup, lower_threshold, upper_threshold, 
                                   lower_threshold+10)
    edged_image = hysteresis(edged_image, lower_threshold+10)

    return edged_image


def main(args):
    """ Driver function for canny edge detection demo
    """
    print("(canny)> Parsing arguments from terminal ...")

    image_path = args.in_file
    output_path = args.out_file

    if os.path.isfile(image_path) == False:
        raise FileNotFoundError("image image not found; {}".format(image_path))

    kernel_size = args.kernel_size    # (int) gaussian smoothing param
    kernel_sigma = args.kernel_sigma  # (float) gaussian smoothing param 
    lower_threshold = args.lower_threshold  # (int) double thresholding params
    upper_threshold = args.upper_threshold  # (int) double thresholding params

    print("-" * 30)
    print("Arguments/params:")
    for arg in vars(args):
        print("\t ..", arg, getattr(args, arg))

    if kernel_size is None:
        print("(canny)> Kernel_size argument not provided, autocompute kernel size from sigma ...")
        kernel_size = compute_gauss_kernel_size(kernel_sigma)

    print("(canny)> Loading image from disk ...")
    image = Image.open(image_path).convert(mode='L')  # read image
    image = np.array(image)  # convert to numpy array

    # run canny edge detection and save
    canny_edge_image = canny_edge_detect(image, kernel_size, kernel_sigma, lower_threshold, upper_threshold)
    plt.imshow(canny_edge_image, cmap='gray')

    print("(canny)> Saving output to %s ..." % output_path)
    plt.imsave(output_path, canny_edge_image, cmap='gray')


# -------------------------------
# DATA LOADING & VIZ
# -------------------------------

def load_set5_dataset(data_dir='./data/'):
    """ Load Image Data
    """

    def load_gray_image(image_path):
        return np.array(Image.open(image_path).convert('L'))
    
    # read and load images from the assignment II, Set5 dataset
    image_cana = load_gray_image(data_dir + '/cana.jpg')
    image_leaf = load_gray_image(data_dir + '/leaf.jpg')
    image_lamp = load_gray_image(data_dir + '/lamp.jpg')
    image_fruit = load_gray_image(data_dir + '/fruit.jpg')
    image_img335 = load_gray_image(data_dir + '/img335.jpg')


def rgb2gray(image):
    # convert to gray image
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])


def plot_set5_images(images, title='NoTitle', imnames=None):
    """Plot images side by side
    """
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    n = len(images)
    imnames = imnames or ([""] * n)

    fig, axes = plt.subplots(2,3, figsize=(20,10))
    for i, image in enumerate(images):
        if len(image.shape) == 3:
            # convert to gray scale first
            image = rgb2gray(image)    

        # plotting image in plot axes
        ax = axes[i%2, i%3]
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(imnames[i])
    
    fig.delaxes(axes[1,2])
    plt.show()  # show set5 images
    plt.savefig('results/set5plot.jpg')


# -------------------------------
# IMAGE OPERATIONS
# -------------------------------

def dnorm(x, mu, sd):
    const_term = 1 / (np.sqrt(2 * np.pi) * sd)
    power_term = np.exp(-np.power((x - mu) / sd, 2) / 2)
    return  const_term * power_term 


def convolution(image, kernel, average=False):
    """ Perform convolution operation for a given kernel
    """
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_height, padded_width = padded_image.shape[:2]
    padded_image[pad_height:padded_height - pad_height, pad_width:padded_width - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output


# -------------------------------
# GAUSSIAN SMOOTHING
# -------------------------------

def compute_gauss_kernel_size(sigma):
    """ Autocompute gaussian kernel size given sigma
    using knowledge about blur radius
    """
    return 2 * math.floor(sigma) + 1


def gaussian_kernel(size, sigma=1):
    """ Generate gaussian smoothing kernel
    """
    kernel_1D = np.linspace(-(size // 2), size // 2, size)

    for i in range(size):
        # calculate gaussian values at row ith
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)

    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)  # compute outer product
    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


def gaussian_smoothing(image, kernel_size):
    """ Perform gaussian smoothing operation
    """
    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size))
    return convolution(image, kernel)


# -------------------------------
# EDGE DETECTION
# -------------------------------

def edge_detection(image, kernel, convert_to_degree=False):
    """ Helper to perform first derivative edge detection
    """
    kernel_y = np.flip(kernel.T, axis=0)

    new_image_x = convolution(image, kernel)
    new_image_y = convolution(image, kernel_y)
    
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    gradient_direction = np.arctan2(new_image_y, new_image_x)

    if convert_to_degree:
        gradient_direction = np.rad2deg(gradient_direction)
        gradient_direction += 180

    return gradient_magnitude, gradient_direction


def sobel_edge_detection(image, convert_to_degree=False):
    """ Perform sobel edge detection
    Return: tuple of (gradient magnitude, edge direction)
    """
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_magnitude, gradient_direction = edge_detection(
        image, sobel_kernel_x, convert_to_degree)

    return gradient_magnitude, gradient_direction


# -------------------------------
# NON-MAX SUPPRESSION
# -------------------------------

PI = 180

def non_max_suppression(gradient_magnitude, gradient_direction):
    """ Apply non-maximum supression for given gradient magnitude and direction
    from first-order derivative edge detection result.
    """
    
    image_row, image_col = gradient_magnitude.shape
    output = np.zeros(gradient_magnitude.shape)

    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]

            # (0 - PI/8 and 15PI/8 - 2PI)
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                prev_pixel = gradient_magnitude[row, col - 1]
                next_pixel = gradient_magnitude[row, col + 1]

            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                prev_pixel = gradient_magnitude[row + 1, col - 1]
                next_pixel = gradient_magnitude[row - 1, col + 1]

            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                prev_pixel = gradient_magnitude[row - 1, col]
                next_pixel = gradient_magnitude[row + 1, col]

            else:
                prev_pixel = gradient_magnitude[row - 1, col - 1]
                next_pixel = gradient_magnitude[row + 1, col + 1]

            if (gradient_magnitude[row, col] >= prev_pixel and
                gradient_magnitude[row, col] >= next_pixel):
                # check if this is maximum edge
                output[row, col] = gradient_magnitude[row, col]

    return output


# -------------------------------
# HYSTERESIS THREHOLDING
# -------------------------------

def double_threshold(image, low, high, weak=50):
    """ Perform double thresholding for image with pixel range [0,255].

    Mark weak edge {low < x < high} with certain value
    Mark strong edge {x > high} with certain value
    """
    output = np.zeros(image.shape)
    strong = 255  # mark strong pixel with this value

    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))

    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    return output


STRONG = 255
PURGED = 0

def hysteresis(edgemap, weak=50):
    """ Perform Hysterisis Thresholding on edged image using iterative methods,
    
    The tracing is performed by iteratively scan into four direction:
    1. Scan top to bottom for each pixel, and look at the neighbouring pixels
    2. Rescan bottom to top for each pixel, and look at the neighbouring pixels
    3. Scan right to left for each pixel, and look at the neighbouring pixels
    4. Re-Scan left to right for each pixel, and look at the neighbouring pixels

    Note: This iterative approach is different from recursive method.
    You can find the recursive approach of hyteresis tracing in C++ version of this code base

    """
    image_row, image_col = edgemap.shape[:2]
    neighbours = [(di, dj) for di in range(-1,2)
                  for dj in range(-1,2)]

    def _check_connected(edgemap, row, col):
        is_connected = [(edgemap[row + di, col + dj] == STRONG)
                        for di, dj in neighbours]
        return any(is_connected)

    # scan edged image from top to bottom
    top_to_bottom = edgemap.copy()
    for row in range(1, image_row):
        for col in range(1, image_col):
            # try to reconnect the weak to strong edge
            if top_to_bottom[row, col] == weak:
                if _check_connected(top_to_bottom, row, col):
                    top_to_bottom[row, col] = STRONG
                else:
                    top_to_bottom[row, col] = PURGED

    # re-scan edged image from bottom to top
    bottom_to_top = edgemap.copy()
    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            # try to reconnect the weak to strong edge
            if bottom_to_top[row, col] == weak:
                if _check_connected(bottom_to_top, row, col):
                    bottom_to_top[row, col] = STRONG
                else:
                    bottom_to_top[row, col] = PURGED

    # re-scan edged image from right to left
    right_to_left = edgemap.copy()
    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            # try to reconnect the weak to strong edge
            if right_to_left[row, col] == weak:
                if _check_connected(right_to_left, row, col):
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    # re-scan edged image from left to right
    left_to_right = edgemap.copy()
    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            # try to reconnect the weak to strong edge
            if left_to_right[row, col] == weak:
                if _check_connected(left_to_right, row, col):
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    final_image[final_image > 255] = 255

    return final_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Perform Canny Edge detection Demo. Callable Python Script")
    
    parser.add_argument('--in-file', '-i', type=str, required=True, 
        help='image file path to the image in .jpg / .jpeg / .png format')
    parser.add_argument('--out-file', '-o', type=str, required=True,
        help='Output file path to save the output edge image, please specify with .jpg /.jpeg / .png')
    parser.add_argument('--kernel-size', '-sz', type=int, default=None, 
        help='Shape of the kernel size. If not specified, will be infered from the sigma.')
    parser.add_argument('--kernel-sigma', '-sig', type=float, default=None, 
        help='Sigma (standard dev) of the gaussian kernel distribution. Will be used also for autocompute kernel size.')
    parser.add_argument('--lower-threshold', '-lt', type=int, default=20, 
        help='Lower threshold (in pixel) for double thresholding')
    parser.add_argument('--upper-threshold', '-ut', type=int, default=100, 
        help='Upper threshold (in pixel) for double thresholding. Must be bigger than lower threshold')

    args = parser.parse_args()

    main(args)

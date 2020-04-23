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


def main(args):
    """ Driver function for canny edge detection demo
    """
    print("(LoG)> Parsing arguments from terminal ...")

    image_path = args.in_file
    output_path = args.out_file

    if os.path.isfile(image_path) == False:
        raise FileNotFoundError("image image not found; {}".format(image_path))

    kernel_size = args.kernel_size    # (int) gaussian smoothing param
    kernel_sigma = args.kernel_sigma  # (float) gaussian smoothing param

    print("-" * 30)
    print("Arguments/params:")
    for arg in vars(args):
        print("\t ..", arg, getattr(args, arg))

    if kernel_size is None:
        print("(LoG)> Kernel_size argument not provided, autocompute kernel size from sigma ...")
        kernel_size = compute_gauss_kernel_size(kernel_sigma)

    print("(LoG)> Loading image from disk ...")
    image = Image.open(image_path).convert(mode='L')  # read image
    image = np.array(image)  # convert to numpy array

    # run laplacian of gaussian edge detection
    zero_crossing_map = gaussian_laplacian_edge_detection(image, kernel_size=kernel_size, sigma=kernel_sigma)
    plt.imshow(zero_crossing_map, cmap='gray')

    # saving output to output path
    print("(LoG)> Saving output to %s ..." % output_path)
    plt.imsave(output_path, zero_crossing_map, cmap='gray')


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
# EDGE DETECTION
# -------------------------------

def compute_gauss_kernel_size(sigma):
    """ Autocompute gaussian kernel size given sigma
    using knowledge about blur radius
    """
    return 2 * math.floor(sigma) + 1


def gaussian_laplace_kernel(size, sigma=1):
    """ Generate Laplacian of Gaussian 2nd derivative operator kernel
    """
    variance = np.power(sigma, 2)
    x_space = np.linspace(-(size//2), size//2, size)

    kernel_1D = x_space.copy()
    for i in range(size):
            kernel_1D[i] = dnorm(x_space[i], 0, sd=sigma)

    # Step 1. kernel_2D is the exponent termb
    # kernel_2D: (1 / (2 pi sigma^2)) . exp((x^2 + y^2) / 2sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    # Step 2. compute the [1 - (x^2 + y^2) / 2sigma^2] term
    xx, yy = np.meshgrid(x_space, x_space)
    middle_term = xx**2 + yy**2
    middle_term = 2 * variance - middle_term 

    # Step 3. kernel 2D is the final gaussian kernel
    # [1 / (pi sigma^4)] . [1 - (x^2 + y^2) / 2sigma^2] . exp[(x^2 + y^2) / 2sigma]
    kernel_2D = -kernel_2D * middle_term / variance**2
    return kernel_2D


def gaussian_laplacian_edge_detection(image, kernel_size=9, sigma=1.4):
    """ Perform Laplacian of Gaussian (LoG) edge detection

    The calculation is broken down as follows:
        1. Generate laplacian of gaussian (LoG) filter
        2. Perform convolution with the (LoG) filter
        3. Computer zero-crossing map and return

    Args:
        .. image (np.array) : image in gray scale to compute
        .. kernel_size (int): size of LoG kernel
        .. sigma (float): standard deviation of gaussian distribution, assumed same for x and y.

    Return: zero crossing map

    """
    neighbours = [(di, dj) for di in range(-1,2)
                  for dj in range(-1,2)]

    def _check_zero_crossing(image, row, col):
        positive_count, negative_count = 0, 0

        for di, dj in neighbours:
            if image[row+di, col+dj] < 0:
                negative_count += 1
            elif image[row+di, col+dj] > 0:
                positive_count += 1
        
        return (negative_count > 0) and (positive_count > 0)

    def _compute_zero_crossing_map(log_image):
        zero_crossing_map = np.zeros(log_image.shape)

        h, w = log_image.shape[:2]

        # perform zero crossing check and compute 
        # zero crossing map
        for row in range(1, h-1):
            for col in range(1, w-1):
                if _check_zero_crossing(log_image, row, col):
                    zero_crossing_map[row, col] = 1
        
        return zero_crossing_map
    
    # Step 1. Calculate LoG kernel or mask
    print("(LoG)> Calculating Laplacian of Gaussian kernel...")
    kernel_2D = gaussian_laplace_kernel(kernel_size, sigma=sigma)

    # Step 2. Perform convolution image with LoG kernel
    print("(LoG)> Performing image convolution with LoG Kernel...")
    log_image = convolution(image, kernel_2D)

    # Step 3. Compute zero crossing map
    print("(LoG)> Computing Zero Crossing map ...")
    zero_crossing_map = _compute_zero_crossing_map(log_image)

    return zero_crossing_map 


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

    args = parser.parse_args()

    main(args)



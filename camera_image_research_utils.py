import numpy as np
import warnings
import random
import cv2
from scipy.ndimage import uniform_filter
from skimage.util import crop

##########
# Common #
##########

# check if image is in grayscale
def is_gray(img):
    """
    Returns if the img is in grayscale
    """
    if len(img.shape) < 2:
        return False
    if len(img.shape) == 2:
        return True
    if img.shape[2]  == 1:
        return True
    if img.shape[2] == 3:
        b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
        if (b==g).all() and (b==r).all():
            return True
    return False

def subtract_running_mean(arr, window_size):
    # Create a window of ones for computing the rolling mean
    window = np.ones(window_size) / window_size
    
    # Compute the rolling mean using convolution
    rolling_mean = np.convolve(arr, window, mode='same')
    
    # Subtract the rolling mean from the original array
    result = arr - rolling_mean
    
    return result

# random crop an input image to certain size
def random_crop(img, size=(200, 200)):
    height = min(img.shape[0], size[0])
    width = min(img.shape[1], size[1])
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


#######################
# Diagonal Extraction #
#######################


def extract_diagonals(image):
    diagonals = []
    antidiagonals = []
    height, width = image.shape
    
    # Extract diagonals
    for d in range(-width + 1, height):
        diagonals.append(np.diagonal(image, offset=d))
    
    # Extract antidiagonals
    for d in range(-width + 1, height):
        antidiagonals.append(np.diagonal(np.flip(image, axis=1), offset=d))
    
    return diagonals, antidiagonals


def diagonals_var_ratio(image):
    image = np.array(image)

    diag, antidiag = extract_diagonals(image)
    diag = np.array([np.var(ele) for ele in diag if len(ele) > 2])
    antidiag = np.array([np.var(ele) for ele in antidiag if len(ele) > 2])

    diag_var = np.nansum(diag)
    antidiag_var = np.nansum(antidiag)

    return np.nan_to_num(diag_var + antidiag_var)


#######################
# Diagonal Extraction #
#######################


def block_reduce_custom(image, func, block_size=(8, 8)):
    image = np.array(image)
    num_rows, num_cols = image.shape[0] // block_size[0], image.shape[1] // block_size[1]
    result = np.empty((num_rows, num_cols), dtype=np.float64)
    
    for i in range(num_rows):
        for j in range(num_cols):
            block = image[i * block_size[0] : (i + 1) * block_size[0],
                          j * block_size[1] : (j + 1) * block_size[1]]
            result[i, j] = func(block)
    
    return result


def high_pass_dct_filtered(img, factor=2):
    """Performing high-pass filtering on the image"""
    img = img.astype(np.float64)
    h, w = img.shape[0], img.shape[1]
    img = img[:h-h%2, :w-w%2]
    if is_gray(img):
        # cater for cases which dimensions are (HxWx1)
        if img.ndim > 2:
            img = np.squeeze(img, axis=2)
        # Apply DCT to the image
        # Convert the image to floating-point data type
        dct_image = cv2.dct(img)
        # Set high-frequency coefficients to zero
        h, w = dct_image.shape
        dct_image[:h//factor, :w//factor] = 0
        # Apply inverse DCT to obtain the modified image
        filtered_image = cv2.idct(dct_image)
        return filtered_image
    else:
        assert img.ndim > 2, f"image shape is invalid: {img.shape}"
        assert img.shape[2] == 3, f"expected img to have 3 channels, but got array with shape {img.shape}"
        temp = []
        # Splitting the color image into separate color channels 
        for idx in range(3):
            channel = img[...,idx]
            # Applying Fourier transform on each color channel
            filtered = high_pass_dct_filtered(channel, factor)
            temp.append(filtered)
        filtered_image = np.stack(temp, axis=-1)
        return filtered_image


#######################
# Codes below Modified from:
# https://github.com/awsaf49/artifact/blob/main/data/transform.py
#######################


#############################################
# Random Crop + JPEG Compression (Modified) #
#############################################


# Configuirations
OUTPUT_SIZE  = 200
CROPSIZE_MIN = 160 # minimum allowed crop size
CROPSIZE_MAX = 2048 # maximum allowed crop size
CROPSIZE_RATIO = (5,8)
QF_RANGE = (65, 100)


def random_crop_resize(img, size=OUTPUT_SIZE):
    """
    This function takes an input image, randomly crops a square region from it, 
    resizes the cropped region to a fixed size, and returns the resulting image.

    img - Image represented in numpy array
    size - Size to which the cropped image will be resized (default is 200x200)
    """
    height, width = img.shape[0], img.shape[1]
    # select the size of crop
    cropmax = min(min(width, height), CROPSIZE_MAX)
    if cropmax < CROPSIZE_MIN:
        warnings.warn("({},{}) is too small".format(height, width))
        return None

    # try to ensure the crop ratio is at least 5/8
    cropmin = max(cropmax * CROPSIZE_RATIO[0] // CROPSIZE_RATIO[1], CROPSIZE_MIN)
    cropsize = random.randint(cropmin, cropmax)

    # select the type of interpolation
    # determines the type of interpolation to use during the resizing step. 
    # It uses cv2.INTER_AREA if the crop size is larger than a constant OUTPUT_SIZE, 
    # otherwise, it uses cv2.INTER_CUBIC.
    interp = cv2.INTER_AREA if cropsize > size else cv2.INTER_CUBIC

    # select the position of the crop
    x1 = random.randint(0, width - cropsize)
    y1 = random.randint(0, height - cropsize)

    # perform the random crop
    cropped_img = img[y1:y1+cropsize, x1:x1+cropsize]

    # perform resizing
    resized_img = cv2.resize(cropped_img, (size, size), interpolation=interp)

    # return the cropped image array
    return resized_img


#######################
# Codes below Referred & Modified from:
# https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/_structural_similarity.py 
#######################


#################################
# Local Correlation Computation #
#################################

def average_local_correlation(im1, im2, win_size=None):
    """
    This functions compute the local correlation between two images patches from two different images. 

    im1         - Image 1
    im2         - Image 2
    win_size    - Window Size
    """
    if win_size == None:
        win_size = min(7, min(im1.shape[0], im1.shape[1]), min(im2.shape[0], im2.shape[1]))
    K2 = 0.03
    ndim = im1.ndim
    NP = win_size ** ndim
    cov_norm = NP / (NP - 1)  # sample covariance
    L = 255 # max pixel values
    C2 = (K2 * L) ** 2

    # Compute (weighted) means
    ux = uniform_filter(im1, size=win_size)
    uy = uniform_filter(im2, size=win_size)

    # Compute (weighted) variances and covariances
    uxx = uniform_filter(im1 * im1, size=win_size)
    uyy = uniform_filter(im2 * im2, size=win_size)
    uxy = uniform_filter(im1 * im2, size=win_size)
    vx = cov_norm * (uxx - ux * ux) # variance of x
    vy = cov_norm * (uyy - uy * uy) # variance of y
    vxy = cov_norm * (uxy - ux * uy) # covariance of x and y

    num = vxy + 0.5*C2
    denom = np.sqrt(np.maximum(vx * vy, 0)) + 0.5*C2

    S = num / denom
    pad = (win_size - 1) // 2

    # Use float64 for accuracy.
    mcorr = crop(S, pad).mean(dtype=np.float64)

    return mcorr
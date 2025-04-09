import numpy as np
from datasets import load_dataset
from camera_image_research_utils import random_crop, random_crop_resize, high_pass_dct_filtered, extract_diagonals, subtract_running_mean, diagonals_var_ratio, block_reduce_custom, average_local_correlation
from PIL import Image
import io

######################
# Feature Generation #
######################

# The following code snippet is the actual code used to generate features for each image in the dataset.
# The example below shows how to generate features for a single image from DiffusionDB dataset.

# Step 1: Load the dataset with the `large_random_1k` subset
dataset = load_dataset('poloclub/diffusiondb', 'large_random_1k')
i = np.random.randint(0, len(dataset['train']))

# Step 2: Preprocess the image
im_ori = np.array(dataset['train'][i]['image'])
im = random_crop(im_ori, (256, 256))

# Step 3: Generate Feature (1)
diag, antidiag = extract_diagonals(high_pass_dct_filtered(im[...,1]))

mean_abs_diag_gradient = np.array([np.mean(np.abs(np.diff(ele.astype(np.uint16)))) for ele in diag if len(ele) > 2])
mean_abs_antidiag_gradient = np.array([np.mean(np.abs(np.diff(ele.astype(np.uint16)))) for ele in antidiag if len(ele) > 2])

# The running averages of the gradient of the diagonals and antidiagonals are subtracted from the original values
demeaned_abs_diag_gradient = subtract_running_mean(mean_abs_diag_gradient, 3)
demeaned_abs_antidiag_gradient = subtract_running_mean(mean_abs_antidiag_gradient, 3)

# The FFT of the demeaned gradient is taken
Y = np.fft.fft(demeaned_abs_diag_gradient)
Y2 = np.fft.fft(demeaned_abs_antidiag_gradient)
abs_Y = np.abs(Y)
abs_Y2 = np.abs(Y2)

# Find where the peaks of the FFTs are
peaks = np.where(abs_Y > np.percentile(abs_Y, 95))
peak1 = np.nan_to_num(np.mean(peaks))
peaks = np.where(abs_Y2 > np.percentile(abs_Y2, 95))
peak2 = np.nan_to_num(np.mean(peaks))

# The proportion of the FFT that is below the peak is calculated
feature1_val1 = (abs_Y < abs_Y[int(peak1)]).mean()
feature1_val2 = (abs_Y2 < abs_Y2[int(peak2)]).mean()

# Step 4: Generate Feature (2)
green_channel = block_reduce_custom(im[...,1], diagonals_var_ratio, block_size=(10, 10))
feature2_val1 = np.sum(green_channel)

red_channel = block_reduce_custom(im[...,0], diagonals_var_ratio, block_size=(10, 10))
feature2_val2 = np.sum(red_channel)

blue_channel = block_reduce_custom(im[...,2], diagonals_var_ratio, block_size=(10, 10))
feature2_val3 = np.sum(blue_channel)

# Step 5: Generate Feature (3)
feature3_val1 = average_local_correlation(green_channel, red_channel) 
feature3_val2 = average_local_correlation(green_channel, blue_channel)
feature3_val3 = average_local_correlation(red_channel, blue_channel)


#####################
# Resize + Compress #
#####################


q = 60
processed_im = random_crop_resize(im_ori, 256)
# Create a virtual file-like object using io.BytesIO
processed_im = Image.fromarray(processed_im)
compressed_image_file = io.BytesIO()
processed_im.save(compressed_image_file, format='JPEG', q=q)
# Open the virtual file-like object using Pillow's Image.open()
read_im = Image.open(compressed_image_file)
# Convert the Pillow image back to a NumPy array
im = np.array(read_im)
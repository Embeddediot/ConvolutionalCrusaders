import torch
import os
from numpy import arange, concatenate

## CONFIG
# define the test split
VAL_SPLIT = 0.15

#NUM workers
NUM_WORKERS = 4

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
 
# define the number of number of classes
NUM_CLASSES = 10

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.5e-3

NUM_EPOCHS = 50
BATCH_SIZE = 16

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

## Dataset & Dataloader
#dir = "drive/MyDrive/Colab Notebooks/02456-deep-learning-with-PyTorch-master/SegCarImg/data/"
dir = r"carseg_data/"

full_range = True

#Find true range
range_1 = arange(1, 501)
range_2 = arange(1300, 2002)
combined_range = concatenate((range_1, range_2))

# Length of images
photo_count = 169 
orange_count = combined_range if full_range else range(1,250) #2001
black_count = 834 if full_range else 250 #834
background_count = 101

# Create a range from 0 to 500 and 1300 to 2001 (this is no noise)

test_images = [dir + 'arrays/photo_{:04d}.npy'.format(i) for i in arange(1,photo_count)]
train_orange_3_doors_images = [dir + 'arrays/orange_3_doors_{:04d}.npy'.format(i) for i in orange_count]
train_black_5_doors_images = [dir + 'arrays/black_5_doors_{:04d}.npy'.format(i) for i in range(1,black_count)]
background_images = [dir + "images/landscapes/{:04d}.jpg".format(i) for i in range(1,background_count)]
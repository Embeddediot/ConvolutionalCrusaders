Car segmentation project

This directory contains the data for the segmenation of car parts by Deloitte. 

The folders:
- images
    - black_5_doors: CAD images of a five door car with black background
    - landscapes: Various landscapes which can be used for adding other backgrounds if engineered for it
    - orange_3_doors: CAD images of a three door OPEL car with orange background
    - photo: Real photos of cars


- arrays: Data saved in numpy format to easily start training models. 
          First 3 channels is the image data 0-255 pixel values. Height, width, RGB.
          Next is target with values described below.

"color",        "description",          "class value"
"orange",       "hood",                 10
"dark green",   "front door",           20
"yellow",       "rear door",            30
"cyan",         "frame",                40
"purple",       "rear quater panel",    50
"light green",  "trunk lid",            60
"blue",         "fender",               70
"pink",         "bumper",               80
"no color",     "rest of car",          90

    - 


###################### TEST DATA ########################
Real car image IDs to use for testing 

TEST IDS:
photo_0001.npy
photo_0002.npy
photo_0003.npy
photo_0004.npy
photo_0005.npy
photo_0006.npy
photo_0007.npy
photo_0008.npy
photo_0009.npy
photo_0010.npy
photo_0011.npy
photo_0012.npy
photo_0013.npy
photo_0014.npy
photo_0015.npy
photo_0016.npy
photo_0017.npy
photo_0018.npy
photo_0019.npy
photo_0020.npy
photo_0021.npy
photo_0022.npy
photo_0023.npy
photo_0024.npy
photo_0025.npy
photo_0026.npy
photo_0027.npy
photo_0028.npy
photo_0029.npy
photo_0030.npy


NOTE: REMEMBER TO DELETE DATA FROM ALL SOURCES FOLLOWING THE PROJECT
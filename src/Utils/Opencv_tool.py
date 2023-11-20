# from skimage import io
# import numpy as np

# arrayz = np.load(trainImages[600])

# image = arrayz[:,:,:3]
# mask = arrayz[:,:,3]

# mask2 = np.where((mask==0),0,1).astype('uint8')
# img = image*mask2[:,:,np.newaxis]

# grayscale_image = np.mean(image, axis=-1)
# print(grayscale_image.shape)
# print(image.shape)
# plt.imshow(grayscale_image,cmap='gray')
# plt.show()


# import cv2
# import numpy as np

# colors_info = {
#     0: (0,0,0),            # background to black 
#     1: (250, 149, 10),     # orange - hood (hMin = 14 , sMin = 213, vMin = 165), (hMax = 19 , sMax = 255, vMax = 255)
#     2: (19, 98, 19),       # dark green - front door (hMin = 57 , sMin = 145, vMin = 63), (hMax = 64 , sMax = 229, vMax = 137)
#     3: (249, 249, 10),     # yellow rear - door (hMin = 27 , sMin = 187, vMin = 126), (hMax = 38 , sMax = 255, vMax = 255)
#     4: (10, 248, 250),     # cyan - frame (hMin = 86 , sMin = 216, vMin = 161), (hMax = 100 , sMax = 255, vMax = 255)
#     5: (149, 7, 149),      # purple - rear quater panel (hMin = 130 , sMin = 236, vMin = 136), (hMax = 151 , sMax = 248, vMax = 164)
#     6: (5, 249, 9),        # light green - trunk lid (hMin = 41 , sMin = 149, vMin = 213), (hMax = 63 , sMax = 255, vMax = 255)
#     7: (20, 19, 249),      # blue - fender (hMin = 117 , sMin = 193, vMin = 181), (hMax = 128 , sMax = 255, vMax = 255)
#     8: (249, 9, 250),      # pink - bumper (hMin = 142 , sMin = 189, vMin = 215), (hMax = 169 , sMax = 255, vMax = 255)
#     9: (255,255,255)       # no color (NA) - rest (hMin = 4 , sMin = 0, vMin = 0), (hMax = 179 , sMax = 76, vMax = 228)
# }

# thredshold = {
#     0: 0,
#     1: 30,
#     2: 17,
#     3: 70,
#     4: 100,
#     5: 10,
#     6: 50,
#     7: 50,
#     8: 50,
#     9: 50
# }

# # Convert RGB values to floats between 0 and 1
# cmap_color = {key: value if isinstance(value, str) else (value[0] / 255, value[1] / 255, value[2] / 255) for key, value in colors_info.items()}
# # Create a custom colormap
# carseg_cmap = ListedColormap(cmap_color.values(), name="custom_cmap")

# img = cv2.imread("carseg_data/images/orange_3_doors/with_segmentation/1300.png")

# # Convert BGR to HSV
# RGB = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ths=0
# mask=0
# for i in range(1,2):
#     colorHSV = print(rgb_to_hsv(colors_info[i][0]/255,
#                           colors_info[i][1]/255,
#                           colors_info[i][2]/255))
#     # lower = np.array(colorHSV)-thredshold[i]
#     # upper = np.array(colorHSV)+thredshold[i]
#     # ths = cv2.inRange(RGB, lower, upper)
#     # ths = np.where(ths==255,i,ths)
#     # mask += ths
#     print(i)

# # display the mask and masked image
# plt.imshow(mask,cmap=carseg_cmap)
# plt.show()
# plt.imshow(RGB)
# plt.show()

# y_process.py
# To be consistent with the paper of ARCNN, only keep the y (out of YCbCr) channel of the original images
#

from __future__ import print_function

from os import listdir
from os.path import join

from PIL import Image
import numpy as np
import h5py 

# from os.pathfrom os.path import join 
'''
ori_train .jpg
ori_val .jpg
->
 ori_y_train .bmp
 ori_y_val .bmp
->
 cmp_y_train .jpg
 cmp_y_val .jpg

'''

input_path = "ori_images/"
input_train_path = join(input_path, "train/")
input_val_path = join(input_path, "val/")

output_path = "ori_y/"
output_train_path = join(output_path, "train/")
output_val_path = join(output_path, "val/")



# y_output_path = "42078_y_test.jpg"


# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in[".png", ".jpg", ".jpeg", ".bmp"])

def convert_to_y_channel(input_path, output_path):
    img_set = listdir(input_path)
    for img_name in img_set:
        img_path = input_path + img_name
        img_num = img_name.split(".")[0]
        img = Image.open(img_path).convert('YCbCr')
        y, _, _ = img.split() 
        y.save(output_path + img_num + ".jpg", "jpeg")
        print(img_name, " convertion success")       
        
# ? what is the size of YCbCr? A: still 0-255
# don't need to convert all images into y-channel-only version. Just convert some images as test.
# well, I guess I need to do so because I need to compress the images after I keep their y channel only

# y, _, _ = test_img.split()
# y.save("42078_y_test.jpg")

# test:
# test_img = Image.open(y_output_path)
# print(test_img)

# np_img = np.asarray(test_img)
# print(np_img)
# print("success!")

convert_to_y_channel(input_train_path, output_train_path)
convert_to_y_channel(input_val_path, output_val_path)


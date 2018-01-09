# h5_data.py


from __future__ import print_function

import os
import Image
import numpy
import h5py
# import math

'''
generate train_data.h5 and test_data.h5

'''
# ori images have only 1 channel, while compressed images have 3 channels
TRAIN_DATA_PATH = "cmp_y/train/"
TRAIN_LABEL_PATH = "ori_y/bmp_ver/train/"

TEST_DATA_PATH = "cmp_y/val/"
TEST_LABEL_PATH = "ori_y/bmp_ver/val/"

output_train_filename = "data_hdf5/train_data.h5"
output_test_filename = "data_hdf5/test_data.h5"

patch_size = 32
stride = 10

def prepare_data(path):
    names = os.listdir(path)
    names = sorted(names) 
    nums = names.__len__()
    
    data = []
    for i in range(nums):
        name = path + names[i]
        print(name)
        img = Image.open(name)

        # img, _, _, = img.split()
        img = numpy.asarray(img)
	# img = img[:, numpy.newaxis]  # is that correct to add the dimension at last?
        size = img.shape
        print(size)
        x_start = 0
        x_end = x_start + patch_size - 1
        y_start = 0
        y_end = y_start + patch_size - 1 
        while x_end < size[0]:
            while y_end < size [1]:
                sub_img = img[x_start:(x_end+1), y_start:(y_end+1)]
                sub_img = sub_img[numpy.newaxis, :, :]
                # notice ":" 
                # print(sub_img.shape)
                data.append(sub_img)

                y_start = y_start + stride
                y_end = y_end + stride
           
            y_start = 0
            y_end = y_start + patch_size - 1
            x_start = x_start + stride
            x_end = x_end + stride
    
    data = numpy.array(data) # notice that list has no shape
    print("data.shape:", data.shape) 

    return data

def write_hdf5(data, label, output_filename):
    # x = data.astype(int) # ? what's the propose of these 2 lines?
    # y = label.astype(int) 
    with h5py.File(output_filename, "w") as h:
        h.create_dataset("data", data=data, shape=data.shape)
        h.create_dataset("label", data=label, shape=label.shape)


if __name__ == "__main__":
    data = prepare_data(TRAIN_DATA_PATH)
    label = prepare_data(TRAIN_LABEL_PATH) 
    write_hdf5(data, label, output_train_filename)

    data = prepare_data(TEST_DATA_PATH)
    label = prepare_data(TEST_LABEL_PATH)    
    write_hdf5(data, label, output_test_filename)
 



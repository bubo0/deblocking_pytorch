#!/bin/bash


train_img_set=`ls ori_y/bmp_ver/train`
val_img_set=`ls ori_y/bmp_ver/val`

# echo files:$files
fileArray=($train_img_set)
# echo total number : ${#fileArray[@]}
# echo the first element :${fileArray[0]}

# echo all elements :
for i in ${fileArray[@]}
do
        echo ${i%.*}
        echo ${i##*.}
        cjpeg -quality 22 -grayscale -outfile cmp_y/train/${i%.*}.jpg ori_y/bmp_ver/train/${i}
done

fileArray2=($val_img_set)
for j in ${fileArray2[@]}
do 
        echo ${j%.*}
        cjpeg -quality 22 -grayscale -outfile cmp_y/val/${j%.*}.jpg ori_y/bmp_ver/val/${j}
done



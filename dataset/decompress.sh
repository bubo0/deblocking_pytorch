#!/bin/bash


files=`ls ori_images/val`
# echo files:$files
fileArray=($files)
# echo total number : ${#fileArray[@]}
# echo the first element :${fileArray[0]}

# echo all elements :
for i in ${fileArray[@]}
do
        echo ${i%.*}
        # echo ${i##*.}
        djpeg -rgb -bmp -outfile ori_images/val_bmp/${i%.*}.bmp ori_images/val/${i}
done

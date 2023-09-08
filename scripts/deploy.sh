#!/bin/bash

torchserve --stop

seg_version=1.0
seg_name="YOLO_MAM"
inp_name="SD_INPAINTING"
inp_version=1.0
files_dir="./torchserve/files"
#
#if [ -e ${files_dir}/src.zip ]; then
#  rm ${files_dir}/src.zip
#fi
#zip -r ${files_dir}/src.zip ./src/
#
#if [ -e ${files_dir}/yolov7.zip ]; then
#  rm ${files_dir}/yolov7.zip
#fi
#zip -r ${files_dir}/yolov7.zip ./yolov7/
#
#if [ -e ${files_dir}/mam.zip ]; then
#  rm ${files_dir}/mam.zip
#fi
#zip -r ${files_dir}/mam.zip ./matting_anything/
#
#if [ -e ${files_dir}/sam.zip ]; then
#  rm ${files_dir}/sam.zip
#fi
#zip -r ${files_dir}/sam.zip ./segment-anything/
#
##### Prepare mar file for segmentation ####
#echo 'Preparing segmentation archive'
## Remove existing mar file
#if [ -e ${files_dir}/${seg_name}.mar ]; then
#  rm ${files_dir}/${seg_name}.mar
#  echo 'Removed existing segmentation archive.'
#fi
## Create mar file
#torch-model-archiver --model-name ${seg_name} \
#                     --version ${seg_version} \
#                     --handler "./torchserve/handlers/yolo_mam_handler.py" \
#                     --extra-files "./model_checkpoints/yolov7-e6e.pt","./model_checkpoints/mam_vitl.pth","./model_checkpoints/sam_vit_l_0b8395.pth","${files_dir}/yolov7.zip","${files_dir}/src.zip","${files_dir}/mam.zip","${files_dir}/sam.zip" \
#                     --export-path ${files_dir}
#                     
#
#
##### Prepare mar file for inpainting ####
#echo 'Preparing inapinting archive'
## Remove existing mar file
#if [ -e ${files_dir}/${inp_name}.mar ]; then
#  rm ${files_dir}/${inp_name}.mar
#  echo 'Removed existing inpainting archive.'
#fi
## Create mar file
#torch-model-archiver --model-name ${inp_name} \
#                     --version ${inp_version} \
#                     --handler "./torchserve/handlers/inpainting_handler.py" \
#                     --extra-files "./model_checkpoints/2401_80e_fine150e.zip","${files_dir}/src.zip" \
#                     --export-path ${files_dir}
#
#### Depolyment ####
torchserve --start --model-store ./torchserve/files/ \
           --models segmentation=${seg_name}.mar inpainting=${inp_name}.mar \
           --ts-config ./config.properties \
           --ncs
      
      
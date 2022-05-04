#!/usr/bin/env bash

# this script resample the image/segmentation nii files to a defined pixel dimension.


# Get a list of patients.
patients=(/Data/McVeighLabSuper/wip/zhennong/nii-images/Abnormal/*/)
patients+=(/Data/McVeighLabSuper/wip/zhennong/nii-images/Normal/*/)
data_folder="/Data/McVeighLabSuper/wip/zhennong/nii-images"
save_folder="/Data/McVeighLabSuper/wip/zhennong/nii-images"
img_or_seg=1 # 1 is image, 0 is seg

if ((${img_or_seg} == 1))
then
img_folder="img-nii"
else
img_folder="seg-pred-0.625-4classes-connected-retouch"
fi

for p in ${patients[*]};
do

# Print the current patient.
  # find out patientclass and patientid
  patient_id=$(basename ${p})
  patient_class=$(basename $(dirname ${p}))

  p=${data_folder}${patient_class}/${patient_id}
  echo ${p} 

  
  # assert whether originial files exist
  if ! [ -d ${p}/${img_folder} ] || ! [ "$(ls -A  ${p}/${img_folder})" ];then
    echo "no image/seg"
    continue
  fi

  
  # set output folder
  
  if ((${img_or_seg} == 1))
  then
  o_dir=${save_folder}${patient_class}/${patient_id}/img-nii-1.5
  else
  o_dir=${save_folder}${patient_class}/${patient_id}/seg-nii-0.625-4classes-connected-retouch-downsample
  fi

  echo ${o_dir}
  mkdir -p ${o_dir}


  IMGS=(${p}/${img_folder}/*.nii.gz)


  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
  do
    i_file=${IMGS[${i}]}
    echo ${i_file}
    o_file=${o_dir}/$(basename ${i_file})

    if [ -f ${o_file} ];then
      echo "already done this file"
      continue
    else
      if ((${img_or_seg} == 1))
      then
        c3d ${i_file} -interpolation Cubic -resample-mm 1.5x1.5x1.5mm -o ${o_file}
      else
        c3d ${i_file} -interpolation NearestNeighbor -resample-mm 1.5mmx1.5mmx1.5mm -o ${o_file}
      fi
    fi   
  done
done



#!/usr/bin/env bash

ins_dir=("concept_data/dog" "concept_data/castle" "concept_data/guitar")     # the path to the reference images 
reg_dir=("./real_reg/samples_dog/" "./real_reg/samples_castle/" "./real_reg/samples_guitar/") # the path to the regularization images 
out_dir=("./logs/dog_1500_rprefix" "./logs/castle_1500_rprefix" "./logs/guitar_1500_rprefix") # output path

ins_prompt=("<new1> dog" "<new1> castle" "<new1> guitar")      # format: "<identifier> <class_noun>"
rep_prompt=("a dog" "a castle" "a guitar")                     # format: "a <class_noun>"
class_noun=("dog" "castle" "guitar")                           # format: "<class_noun>"

max_steps=(1500 1500 1500)

length=${#ins_dir[@]}

#python src/retrieve.py --target_name "${ARRAY[0]}" --outpath ${ARRAY[2]}

i=0
while [ $i -lt $length ]; do

    accelerate launch --num_processes 2 src/diffusers_training_sdxl.py \
          --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
          --instance_data_dir="${ins_dir[$i]}"  \
          --class_data_dir="${reg_dir[$i]}" \
          --output_dir="${out_dir[$i]}"  \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --instance_prompt="${ins_prompt[$i]}"  \
          --replace_prompt="${rep_prompt[$i]}" \
          --class_prompt="${class_noun[$i]}" \
          --resolution=512  \
          --train_batch_size=1  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=${max_steps[$i]} \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"  \
          --mixed_precision bf16 \
          --seed 42

    i=$((i + 1))
done
#!/bin/bash
printf "1. Use randomly-generated preference matrix.\n2. Use preference matrix from LTR dataset.\n"
printf "Select: "
read dataset
printf "How many arms are there?\n"
printf "Input: "
read num_arms
if [ $dataset -eq 1 ]
then
  printf "What is the effect size?\n"
  printf "Input: "
  read effect_size
else
  printf "The effect size is set in the real-world dataset.\n"
fi
printf "How many simulations do you want?\n"
printf "Input: "
read num_simulations

if [ $dataset -eq 1 ]
then
  nohup python3 dataset_generator.py RANDOM $effect_size $num_arms $num_simulations &
else
  nohup python3 dataset_generator.py LTR $num_arms $num_simulations &
fi
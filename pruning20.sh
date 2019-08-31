#!/bin/bash
#--k 1.5 --print_freq 200 --HTrate 0.9

change_layer_end_for_different_structure(){
resnet110 324
resnet56 162
resnet32 90 
resnet20 54
}
 
pruning(){
CUDA_VISIBLE_DEVICES=$1 python cifar10_resnet_test.py  data --dataset cifar100 --arch resnet110  --depth 110 \
--save_path $2 \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--print_freq 200  #  --resume ./save_100/cifar100_resnet56_0.9_norm.pth.tar
}


pruning 1 ./baseline/cifar100_resnet110_re_2

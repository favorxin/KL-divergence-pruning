# KL-divergence-pruning
We pruning deep neural networks via Kullback-Leibler Divergence

# Requirements
Python3.6
Pytorch0.4.0

# Introduction
We modify the code on https://github.com/Eric-mingjie/rethinking-network-pruning. We adopt three-stages pruning method(training, pruning, 
fine-tuning).

# The pruned models
You can find the pruned model in 链接：https://pan.baidu.com/s/1LzpkpDPsdJ6iVKiz6TDxnA 提取码：p5oj .
We use dataset+network+pruning rate+methods to represent the pruned model.

# Train the orginal well-pretrained network.
'''
sh pruning20.sh
'''
# Pruning the model
'''
python res56prune.py --dataset cifar10 -v A --model ./logs/cifar10_resnet56_baseline_re/best.resnet56.pth.tar
--save ./logs --test-batch-size 256 --depth 56 -type kl
'''

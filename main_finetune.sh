CUDA_VISIBLE_DEVICES=0 python main_finetune.py --refine ./logs/cifar100_resnet110_0.8_kl_1.pth.tar --dataset cifar100 --arch resnet --depth 110 --epoch 200 --save_path ./resnet_100/cifar100_resnet110_0.8_kl_3  --lr 0.001  --schedule 1 60 120 160 --gammas 10 0.2 0.2 0.2

# python main.py --dataset cifar10 --arch resnet --depth 110 --save ./logs/cifar10_resnet110/

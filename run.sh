CUDA_VISIBLE_DEVICES=0 nohup python Train_cifarn.py --batch_size 256 --noise_type aggre --num_epochs 420 --lr 0.05 --cosine\
 --data_path ./cifar-10 --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.99\
 --pretrain_ep 0 --start_expand 100 > c10_aggre.log &  


CUDA_VISIBLE_DEVICES=0 nohup python Train_cifarn.py --batch_size 256 --noise_type worst --num_epochs 420 --lr 0.05 --cosine\
 --data_path ./cifar-10 --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.99\
 --pretrain_ep 0 --start_expand 100 > c10_worst.log &  



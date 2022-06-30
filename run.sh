CUDA_VISIBLE_DEVICES=0 nohup python Train_cifarn.py --batch_size 256 --noise_type aggre --num_epochs 420 --lr 0.05 --cosine\
 --data_path ./cifar-10 --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.99\
 --pretrain_ep 10 --start_expand 150 > c10_aggre.log &  

CUDA_VISIBLE_DEVICES=0 nohup python Train_cifarn.py --batch_size 256 --noise_type rand1 --num_epochs 420 --lr 0.05 --cosine\
 --data_path ./cifar-10 --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.99\
 --pretrain_ep 10 --start_expand 150 > c10_rand1.log &  

CUDA_VISIBLE_DEVICES=0 nohup python Train_cifarn.py --batch_size 256 --noise_type worst --num_epochs 420 --lr 0.05 --cosine\
 --data_path ./cifar-10 --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.99\
 --pretrain_ep 10 --start_expand 150 > c10_worst.log &  


CUDA_VISIBLE_DEVICES=0 nohup python Train_cifarn.py --batch_size 256 --noise_type noisy100 --num_epochs 420 --lr 0.05 --cosine\
 --data_path ./cifar-100 --dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --threshold 0.9 --tau 0.95\
 --pretrain_ep 30 --start_expand 150 > c10_worst.log &  

# For evluation (after running the above shells)
python learning.py --dataset cifar10 --noise_type aggre --val_ratio 0.1 

python learning.py --dataset cifar10 --noise_type rand1 --val_ratio 0.1 

python learning.py --dataset cifar10 --noise_type worst --val_ratio 0.1 

python learning.py --dataset cifar100 --noise_type noisy100 --val_ratio 0.1 

# For detection (after running the above shells)
python detection.py --dataset cifar10 --noise_type aggre --val_ratio 0.1 

python detection.py --dataset cifar10 --noise_type rand1 --val_ratio 0.1 

python detection.py --dataset cifar10 --noise_type worst --val_ratio 0.1 

python detection.py --dataset cifar100 --noise_type noisy100 --val_ratio 0.1 

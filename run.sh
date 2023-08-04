# For CIFAR-10-Symmetric
python Train_promix.py --cosine --dataset cifar10 --num_class 10 --rho_range 0.7,0.7 --tau 0.99 --pretrain_ep 10 --debias_output 0.8 --debias_pl 0.8  --noise_mode sym --noise_rate 0.2
python Train_promix.py --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10 --debias_output 0.8 --debias_pl 0.8  --noise_mode sym --noise_rate 0.5

# For CIFAR-10-Asymmetric
python Train_promix.py --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10 --debias_output 0.8 --debias_pl 0.8  --noise_mode asym --noise_rate 0.4

# For CIFAR-100-Symmetric
python Train_promix.py --cosine --dataset cifar100 --num_class 100 --rho_range 0.7,0.7 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5  --noise_mode sym --noise_rate 0.2
python Train_promix.py --cosine --dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5  --noise_mode sym --noise_rate 0.5

# For CIFAR-10N 
#aggre
python Train_promix.py --noise_type aggre --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10  --noise_mode cifarn
#rand1
python Train_promix.py --noise_type rand1 --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10  --noise_mode cifarn
#worst
python Train_promix.py --noise_type worst --cosine --dataset cifar10 --num_class 10 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 10  --noise_mode cifarn

# For CIFAR-100N
python Train_promix.py --noise_type noisy100 --cosine --dataset cifar100 --num_class 100 --rho_range 0.5,0.5 --tau 0.95 --pretrain_ep 30 --debias_output 0.5 --debias_pl 0.5 --noise_mode cifarn


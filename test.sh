CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --resume \
/scratch/adityaRRC/logs/run/iiitds_image_only/iiitds/deeplab-drn/experiment_0/checkpoint_25_.pth.tar \
--logsFlag iiitds_image_only --batch-size 16 --epoch 1 --mode test --dataset \
iiitds  

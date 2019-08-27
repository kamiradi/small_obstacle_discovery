CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --resume \
/scratch/adityaRRC/logs/run/Lidar_depth/iiitds/deeplab-drn/experiment_0/model_best.pth.tar \
--batch-size 2 --epoch 1 --mode test --dataset iiitds --depth --logsFlag \
Lidar_depth/val

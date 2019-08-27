CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --resume \
/scratch/adityaRRC/logs/run/iiitds_image_only/iiitds/deeplab-drn/experiment_1/model_best.pth.tar \
--logsFlag iiitds_image_only_val --batch-size 2 --epoch 1 --mode test --dataset \
iiitds 

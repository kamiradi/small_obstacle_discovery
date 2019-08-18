CUDA_VISIBLE_DEVICES=3 python train.py --depth_path \
    deeplab-small_obs-depth_input_heteroscedastic.pth \
    --image_path deeplab-small_obs-image_input_heteroscedastic.pth \
    --batch-size 2 --epoch 40 --mode train --dataset lnf --gpu-ids 0 --depth \
    --logsFlag twindeeplab 

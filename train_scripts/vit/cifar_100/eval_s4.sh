CUDA_VISIBLE_DEVICES=0,  python -m torch.distributed.launch --nproc_per_node=1  --master_port=12346  \
	train.py /path/to/cifar100 --dataset torch/cifar100 --num-classes 100 --model vit_base_patch16_224_in21k  \
    --batch-size 1 \
    --drop-path 0 --img-size 224 \
	--output  output/vit_base_patch16_224_in21k/cifar_100/vit_hsn_eval \
	--amp  --pretrained \
    --evaluate \
    --initial-checkpoint path/to/model_best.pth.tar \


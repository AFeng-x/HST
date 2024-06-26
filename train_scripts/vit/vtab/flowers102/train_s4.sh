CUDA_VISIBLE_DEVICES=3,4,  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=14223  \
	train.py /path/to/vtab-1k/oxford_flowers102 --dataset flowers102 --num-classes 102  --no-aug --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-5 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vtab/flowers102/vit_hsn \
	--amp  --pretrained  

# lr 1e-3  wd 5e-5  dp 0.0

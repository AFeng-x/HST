CUDA_VISIBLE_DEVICES=3,4,  python  -m torch.distributed.launch --nproc_per_node=2  --master_port=14332  \
	train.py /path/to/vtab-1k/clevr_count --dataset clevr_count --num-classes 8  --no-aug  --direct-resize --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-4 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0.1 --img-size 224 \
	--mixup 0 --cutmix 0 --smoothing 0 \
	--output  output/vtab/clevr_count/vit_hsn \
	--amp --pretrained  \

# lr 1e-3 --weight 5e-4 dp 0.1
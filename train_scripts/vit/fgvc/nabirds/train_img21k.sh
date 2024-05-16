CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4 --master_port=14223  \
	train.py /path/to/nabirds --dataset nabirds --num-classes 555  --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 5e-2 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 5e-4 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  output/vit_hsn/fgvc/nabirds/img21k \
	--amp --pretrained

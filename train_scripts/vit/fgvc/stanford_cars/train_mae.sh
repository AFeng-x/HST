CUDA_VISIBLE_DEVICES=0,1,2,3,  python  -m torch.distributed.launch --nproc_per_node=4  --master_port=12349  \
	train.py /path/to/stanford_cars --dataset stanford_cars --num-classes 196 --val-split val --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 32 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-3 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  output/vit_hsn/fgvc/stanford_cars/mae \
	--amp --initial-checkpoint /path/to/mae_pretrain_vit_base.pth

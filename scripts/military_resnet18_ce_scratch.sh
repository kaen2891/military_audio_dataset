MODEL="resnet18"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs64_lr1e-3_ep400_seed${s}_scratch"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset military \
                                        --seed $s \
                                        --n_cls 7 \
                                        --epochs 400 \
                                        --batch_size 64 \
                                        --optimizer adam \
                                        --learning_rate 1e-3 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --model $m \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --method ce \
                                        --print_freq 100

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_ast_ce_bs8_lr5e-5_ep50_seed1/best.pth

    done
done


TASK=binary_class_task
ARCH=drug_gcn_large
CLSHEAD=bclsmlpmixSimplifiedV2Diff
CRITERION=binary_class_loss_cons_neg_sigmoid
DATAFOLD=$1
ALPHA=$2
LR=$3
BATCH_SIZE=$4

DATADIR=/blob2/v-jiaclin/dmp/data/twosides/$DATAFOLD/data-bin
SAVEDIR=/blob2/v-jiaclin/code/dmp/ckpt/twosides/bib_ddi_ckpt/ddi_zoo/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-alpha$ALPHA-batch_size$BATCH_SIZE-v3-max

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=0 python ddi_zoo/src/train.py $DATADIR \
    --user-dir ddi_zoo/src/ \
    --tensorboard-logdir $SAVEDIR \
    -s 'a' -t 'b' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size $BATCH_SIZE \
    --required-batch-size-multiple 1 \
    --classification-head-name $CLSHEAD \
    --num-classes 963 \
    --pooler-dropout 0.1 \
    --reg-loss-weight $ALPHA \
    --gnn-norm layer \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update 1000000 \
    --warmup-updates 70000 --max-update 1000000 \
    --log-format 'simple' --log-interval 100 \
    --fp16 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \

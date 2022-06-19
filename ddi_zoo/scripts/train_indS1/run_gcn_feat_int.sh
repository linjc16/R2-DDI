
TASK=inductive_task
ARCH=drug_gcn_large
CLSHEAD=bclsFeatInt
CRITERION=binary_class_loss
DATAFOLD=$1
LR=$2
BATCH_SIZE=$3

DATADIR=/home/v-jiaclin/blob2/v-jiaclin/dmp/data/new_split/ind_unseen/$DATAFOLD/data-bin
SAVEDIR=/home/v-jiaclin/blob2/v-jiaclin/code/dmp/ckpt/bib_ddi_ckpt_S1/ddi_zoo/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-batch_size$BATCH_SIZE

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
    --num-classes 86 \
    --pooler-dropout 0.1 \
    --gnn-norm layer \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update 50000 \
    --warmup-updates 4000 --max-update 50000 \
    --log-format 'simple' --log-interval 100 \
    --fp16 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \

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

CUDA_VISIBLE_DEVICES=0 python ddi_zoo/src/binary_class_inf.py $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir ddi_zoo/src/ \
    --ddp-backend=legacy_ddp \
    --reset-dataloader \
    -s 'a' -t 'b' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size 512 \
    --optimizer adam \
    --gnn-norm layer \
    --classification-head-name $CLSHEAD \
    --num-classes 963 \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'valid' \


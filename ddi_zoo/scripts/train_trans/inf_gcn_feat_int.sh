
TASK=binary_class_task
ARCH=drug_gcn_large
CLSHEAD=bclsFeatInt
CRITERION=binary_class_loss
DATAFOLD=$1
LR=$2

DATADIR=/home/v-jiaclin/blob2/v-jiaclin/dmp/data/new_split/transductive/$DATAFOLD/data-bin
SAVEDIR=/home/v-jiaclin/blob2/v-jiaclin/code/dmp/ckpt/bib_ddi_ckpt/ddi_zoo/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR

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
    --num-classes 86 \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'test' \

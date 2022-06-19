
# binarize dataset

DATADIR=data/new_split/inductive/new_build3

fairseq-preprocess \
    -s 'a' -t 'b' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --destdir  $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

fairseq-preprocess \
    -s 'nega' -t 'negb' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --destdir $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

fairseq-preprocess \
    --only-source \
    --trainpref $DATADIR/train.label \
    --validpref $DATADIR/valid.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers 30 --srcdict $DATADIR/label.dict \

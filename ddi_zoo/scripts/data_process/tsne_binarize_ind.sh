
# binarize dataset

DATADIR=data/tsne/

fairseq-preprocess \
    -s 'a' -t 'b' \
    --validpref $DATADIR/valid \
    --destdir  $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

fairseq-preprocess \
    -s 'nega' -t 'negb' \
    --validpref $DATADIR/valid \
    --destdir $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

fairseq-preprocess \
    --only-source \
    --validpref $DATADIR/valid.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers 30 --srcdict $DATADIR/label.dict \

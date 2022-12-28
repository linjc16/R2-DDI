# R<sup>2</sup>-DDI: Relation-aware Feature Refinement for Drug-drug Interaction Prediction
This repository is the official implementation of [R<sup>2</sup>-DDI: Relation-aware Feature Refinement for Drug-drug Interaction Prediction](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac576/6961471?utm_source=authortollfreelink&utm_campaign=bib&utm_medium=email&guestAccessKey=189b0995-bc41-40fc-b625-bf34b44ff21e&login=true). The code is originally forked from [Fairseq](https://github.com/pytorch/fairseq) and [DVMP](https://github.com/microsoft/DVMP).

## Requirements and Installation
* PyTorch version == 1.8.0
* PyTorch Geometric version == 1.6.3
* RDKit version == 2020.09.5

You can build the [Dockerfile](Dockerfile) or use the docker image `teslazhu/pretrainmol36:latest`.

To install the code from source
```
git clone https://github.com/linjc16/R2-DDI.git

pip install fairseq
pip uninstall -y fairseq 

pip install ninja
python setup.py build_ext --inplace
```
## Getting Started
### Dataset
The raw dataset can be downloaded [here](https://bitbucket.org/kanz76/data-collection/src/master/DDI/). Just check `preprocessed` folder (all the negative samples are generated and saved already).

### Data Preprocessing
We evaluate our models on DrugBank and TwoSides benchmark sets. `ddi_zoo/scripts/data_process` and `ddi_zoo/scripts/twosides/data_process` are folders for preprocessing of DrugBank and TwoSides, respectively. To generate the binary data for `fairseq`, take the transductive setting for DrugBank as an example, run
```
python ddi_zoo/scripts/data_process/split_trans.py

python ddi_zoo/scripts/data_process/run_process_trans.py

bash ddi_zoo/scripts/data_process/run_binarize_trans.sh
```

Note that you need to change the file paths accordingly.

## Training and Test
All traning and test scripts can be seen in `ddi_zoo/scripts`. For instance,
```
bash ddi_zoo/scripts/train_trans/run_gcn_feat_int_cons.sh new_build1 0.01 1e-4 256

bash ddi_zoo/scripts/train_trans/inf_gcn_feat_int_cons.sh new_build1 0.01 1e-4 256
```
## Contact
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.

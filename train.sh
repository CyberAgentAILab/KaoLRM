set -e

ACC_CONFIG="./configs/accelerate-train.yaml"
TRAIN_CONFIG="./configs/train-multiview-base.yaml"
# TRAIN_CONFIG="./configs/train-mono-base.yaml"

PYTHONWARNINGS=ignore accelerate launch --config_file $ACC_CONFIG -m kaolrm.launch train.lrm --config $TRAIN_CONFIG

INFER_CONFIG="./configs/infer-base-mono.yaml"
MODEL_NAME="./releases/mono/"
IMAGE_INPUT="data/sample_input"

python -m kaolrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT 

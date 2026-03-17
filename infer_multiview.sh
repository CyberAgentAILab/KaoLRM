INFER_CONFIG="./configs/infer-base-multiview.yaml"
MODEL_NAME="./releases/multiview/"
IMAGE_INPUT="data/sample_input"

python -m kaolrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT 

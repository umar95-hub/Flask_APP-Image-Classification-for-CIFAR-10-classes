import os
from model_1 import get_model, pred
from utils import pre_process

filename = "Screenshot 2023-03-09 153454.jpg"
img_address = os.path.join("/config/workspace/images", filename)

image_tensor = pre_process(img_address)
tflite_model, input_details, output_details = get_model()

result = pred(image_tensor,tflite_model, input_details, output_details )

print(result)
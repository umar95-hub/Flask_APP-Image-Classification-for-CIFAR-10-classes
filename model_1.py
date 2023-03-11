# Import TensorFlow Lite
import tflite_runtime.interpreter as tflite
import numpy as np

import logging

#logging.basicConfig(filename='logger_task_1.txt',level = logging.DEBUG,'%(asctime)s %(message)s')

logging.basicConfig(filename='logger_task_1.txt',level = logging.INFO,format = '%(levelname)s %(asctime)s %(message)s')

#################################### Model #################################
def get_model():
    # Load a TensorFlow Lite model
    tflite_model = tflite.Interpreter(model_path='my_model.tflite')
    tflite_model.allocate_tensors()

    # Get input and output tensors
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()

    return tflite_model, input_details, output_details

def pred(image_tensors, tflite_model, input_details, output_details):
    # Run inference on the TensorFlow Lite model
    classes = {0: 'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

    tflite_model.set_tensor(input_details[0]['index'],image_tensors)
    tflite_model.invoke()
    tflite_pred = tflite_model.get_tensor(output_details[0]['index'])

    pred_score = np.max(tflite_pred)
    pred_class = classes[int(np.argmax(tflite_pred ))]

    return  pred_class,pred_score
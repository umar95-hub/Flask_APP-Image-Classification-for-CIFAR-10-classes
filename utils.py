import numpy as np
from PIL import Image
import logging

#logging.basicConfig(filename='logger_task_1.txt',level = logging.DEBUG,'%(asctime)s %(message)s')

logging.basicConfig(filename='logger_utils.txt',level = logging.INFO,format = '%(levelname)s %(asctime)s %(message)s')

def pre_process(img_address):
    logging.info(f"Input file path: {img_address}")
    
    try: 
        im1 = Image.open(img_address)

    except Exception as e:

        print("Please input a valid image ")

        return 

    new_image = im1.resize((32,32))
    new_image = np.array(new_image)/255
    new_image = new_image.astype('float32')

    new_image = np.expand_dims(new_image,axis= 0)

    try: 
        new_image = new_image[:,:,:,:3]
    except :
        print("Enter a RGB image")
    # Output Images
    #plt.imshow(new_image)
    return new_image
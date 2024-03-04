import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt 

# Define our imshow function 
def imshow(title="Image", image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

# Set up relative paths for the models, output folder, and the art directory
model_file_path = 'NeuralStyleTransfer\models'  # Relative path for models
output_folder = 'output'  # Relative path for output
art_path = 'NeuralStyleTransfer\art'  # Relative path for art images
img_path = "input_images\city.jpg"  # Example image path


# Load image
img = cv2.imread(img_path)

model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]

# Loop through and apply each model style to our input image
for (i, model) in enumerate(model_file_paths):
    print(str(i+1) + ". Using Model: " + model[:-3])
    style = cv2.imread(join(art_path, model[:-3] + ".jpg"))
    # Load neural style transfer model
    neuralStyleModel = cv2.dnn.readNetFromTorch(join(model_file_path, model))

    # Resize image
    height, width = img.shape[:2]
    newWidth = int((640 / height) * width)
    resizedImg = cv2.resize(img, (newWidth, 640), interpolation=cv2.INTER_AREA)

    # Create blob and perform a forward pass
    inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640), (103.939, 116.779, 123.68), swapRB=False, crop=False)
    neuralStyleModel.setInput(inpBlob)
    output = neuralStyleModel.forward()

    # Post-process output
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.68
    output /= 255
    output = output.transpose(1, 2, 0)
    output = (output * 255).astype(np.uint8)
    
    # Save the output
    output_filename = f"output_{i+1}.png"
    cv2.imwrite(join(output_folder, output_filename), output)
    # Optionally display the result with imshow function
    # imshow("Neural Style Transfers", output, size=10)

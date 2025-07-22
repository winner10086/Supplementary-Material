import cv2
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from scipy.ndimage import median_filter, gaussian_filter
from torchvision.models import alexnet
import numpy as np
import torch.nn as nn


def calculate_gradient(image_path, label, weight_path):
    # Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = data_transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    # Load model
    model = alexnet(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Define loss
    loss = nn.CrossEntropyLoss()
    label_tensor = torch.tensor([label], dtype=torch.long).to(device)

    # Forward pass
    outputs = model(img_tensor)

    # Calculate loss
    cost = loss(outputs, label_tensor)

    # Calculate gradient
    grad = torch.autograd.grad(cost, img_tensor, retain_graph=False, create_graph=False)[0]

    return grad.squeeze(0).cpu().numpy()

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

# Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    # Load model weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()
    return(predicted_class_index,output[index])

def divide_matrix(input_matrix, level):
    unique_elements = np.unique(input_matrix)
    num_unique = len(unique_elements)
    threshold = np.percentile(unique_elements, level)
    mark_matrix_actual_index = np.where(input_matrix > threshold, 1, 0)
    return mark_matrix_actual_index

###########################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# The absolute path of index file
index_file_absolute_path = r"..."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# The absolute path of weight file for convolutional neural network model
weight_file_absolute_path = r"..."

model_current = alexnet

# Import the folder path where the original input images are stored.
actual_image_absolute_path = "..."

image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
actual_image = np.array(image)

# The R, G, B three channel matrix of the actual image
actual_image_channel_R = actual_image[:, :, 0]
actual_image_channel_R = actual_image_channel_R.astype(np.float64)
actual_image_channel_G = actual_image[:, :, 1]
actual_image_channel_G = actual_image_channel_G.astype(np.float64)
actual_image_channel_B = actual_image[:, :, 2]
actual_image_channel_B = actual_image_channel_B.astype(np.float64)

# Standardize the actual image
actual_image_transform_matrix_R = ((actual_image_channel_R / 255) - 0.485) / 0.229
actual_image_transform_matrix_G = ((actual_image_channel_G / 255) - 0.456) / 0.224
actual_image_transform_matrix_B = ((actual_image_channel_B / 255) - 0.406) / 0.225
####################################################################################################
# Perform category prediction on the original image
actual_image_index, x = predict_image_path(actual_image_absolute_path, index_file_absolute_path,
                                       weight_file_absolute_path, 0, model_current)
print(f"Top-1 category index number of the actual image:{actual_image_index}")
####################################################################################
gradient_matrix = -1 * calculate_gradient(actual_image_absolute_path, actual_image_index, weight_file_absolute_path)
iterative_image_pixel_weight_matrix_R = gradient_matrix[0]
iterative_image_pixel_weight_matrix_G = gradient_matrix[1]
iterative_image_pixel_weight_matrix_B = gradient_matrix[2]

# Calculate the total pixel classification contribution matrix
pixel_classification_contribution_matrix_total = np.abs(iterative_image_pixel_weight_matrix_R) + np.abs(iterative_image_pixel_weight_matrix_G) + np.abs(iterative_image_pixel_weight_matrix_B)

pixel_classification_contribution_matrix_total_plot = pixel_classification_contribution_matrix_total.copy()

smoothed = median_filter(pixel_classification_contribution_matrix_total, size=3)
smoothed = gaussian_filter(smoothed, sigma=20)
pixel_classification_contribution_matrix_total = smoothed.copy()

flattened = pixel_classification_contribution_matrix_total.flatten()
sorted_indices = np.argsort(flattened)
ranks = np.empty_like(sorted_indices)
ranks[sorted_indices] = np.arange(1, len(flattened) + 1)
pixel_classification_contribution_matrix_total = ranks.reshape(pixel_classification_contribution_matrix_total.shape)

pixel_classification_contribution_matrix_total = (pixel_classification_contribution_matrix_total - pixel_classification_contribution_matrix_total.min()) / (pixel_classification_contribution_matrix_total.max() - pixel_classification_contribution_matrix_total.min() + 1e-8)

img = Image.open(actual_image_absolute_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., pixel_classification_contribution_matrix_total, use_rgb=True)
image = Image.fromarray((visualization).astype(np.uint8))
image.save("saliency.png")
img = Image.open(actual_image_absolute_path).convert('RGB')
img = np.array(img, dtype=np.uint8)
input_img = img.astype(dtype=np.float32) / 255.

mark_matrix = divide_matrix(pixel_classification_contribution_matrix_total, 95)

zero_positions = np.where(mark_matrix == 0)

image_path = actual_image_absolute_path
image = Image.open(image_path)
image_array = np.array(image)

image_array[zero_positions[0], zero_positions[1], :] = 255

output_image = Image.fromarray(image_array)

for i in range(224):
    for j in range(224):
        if mark_matrix[i][j] == 0:
            pixel_classification_contribution_matrix_total[i][j] = 0

visualization = show_cam_on_image(input_img, pixel_classification_contribution_matrix_total, use_rgb=True)

overlay_image = Image.fromarray((visualization).astype(np.uint8))
overlay_image.save("saliency map.png")

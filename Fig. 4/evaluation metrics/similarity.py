import os
import numpy as np
from PIL import Image

def image_to_array(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    return np.array(img, dtype=np.int16)

# Import the folder path where the original input images are stored.# Import the folder path where the original input images are stored.
folder1 = r"..."

# Import the path of the folder where the adversarial examples are stored.
folder2 = r"..."

image_names = [f for f in os.listdir(folder1) if f.endswith('.png')]
image_names.sort(key=lambda x: int(x.split('.')[0]))

total_pixel = 224 * 224 * 3

scores = []

number_total = 0
intensity_total = 0

num_images = 0

for name in image_names:
    num_images = num_images + 1
    path1 = os.path.join(folder1, name)
    path2 = os.path.join(folder2, name)

    img1 = image_to_array(path1)
    img2 = image_to_array(path2)

    diff_mask = (img1 != img2)
    a = np.count_nonzero(diff_mask)

    max_diff = np.max(np.abs(img1 - img2))
    b = max_diff

    norm_a = a / total_pixel
    norm_b = b / 255

    number_total = number_total + norm_a
    intensity_total = intensity_total + norm_b

    score = np.sqrt(norm_a ** 2 + norm_b ** 2)
    scores.append(score)

final_score = np.mean(scores)
print("The average value of the similarity of all adversarial examples is:", final_score)

print(f"The average number of tampered pixels is: {number_total / num_images}")
print(f"The average tampering intensity is: {intensity_total / num_images}")
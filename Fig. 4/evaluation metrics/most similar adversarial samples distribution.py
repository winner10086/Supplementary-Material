import os
import numpy as np
from PIL import Image
from collections import defaultdict

def image_to_array(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    return np.array(img, dtype=np.int16)

# Import the folder path where the original input images are stored.
original_folder = r"..."

# Import the path of the folder where the adversarial examples are stored.
root_folder = r"..."

image_names = [f for f in os.listdir(original_folder) if f.endswith('.png')]
image_names.sort(key=lambda x: int(x.split('.')[0]))

subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
subfolders.sort(key=lambda x: float(x))
print(subfolders)

total_pixel = 224 * 224 * 3

best_count = defaultdict(int)
best_ratios = {}

for name in image_names:
    orig_path = os.path.join(original_folder, name)
    orig_img = image_to_array(orig_path)

    best_score = float('inf')
    best_ratio = None

    for folder in subfolders:
        folder_path = os.path.join(root_folder, folder)
        tampered_path = os.path.join(folder_path, name)

        if not os.path.exists(tampered_path):
            continue

        tampered_img = image_to_array(tampered_path)

        diff_mask = (orig_img != tampered_img)
        a = np.count_nonzero(diff_mask)
        b = np.max(np.abs(orig_img - tampered_img))

        norm_a = a / total_pixel
        norm_b = b / 255
        score = np.sqrt(norm_a ** 2 + norm_b ** 2)

        if score < best_score:
            best_score = score
            best_ratio = folder

    if best_ratio is not None:
        best_count[best_ratio] += 1
        best_ratios[name] = best_ratio

total_images = len(image_names)
ratio_distribution = {k: round(v / total_images, 4) for k, v in best_count.items()}

for ratio in sorted(subfolders, key=lambda x: float(x)):
    count = best_count.get(ratio, 0)
    percent = (count / total_images) * 100
    print(f"Tampering ratio {ratio}: optimal image count {count}, accounting for {percent:.2f}%")

print("\nOptimal tampering ratio corresponding to each image:")
for img_name, ratio in best_ratios.items():
    print(f"Image {img_name}: optimal tampering ratio {ratio}")
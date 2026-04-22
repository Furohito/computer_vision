import os
import numpy as np
from PIL import Image

input_folder = "hakim"

label_map = {
    "circle": 0,
    "diamond": 1,
    "ellipse_h": 2,
    "ellipse_v": 3,
    "parallelogram": 4,
    "rectangle_h": 5,
    "rectangle_v": 6,
    "square": 7,
    "trapezium": 8,
    "triangle": 9
}

X_data = []
y_data = []

for file in os.listdir(input_folder):
    full_path = os.path.join(input_folder, file)

    if not os.path.isfile(full_path):
        continue

    filename = file.lower()

    label = None
    for key in label_map:
        if key in filename:
            label = label_map[key]
            break

    if label is None:
        continue

    img = Image.open(full_path).convert("L")
    img_array = np.array(img) / 255.0

    flat = img_array.flatten()

    X_data.append(flat)
    y_data.append(label)

X = np.array(X_data)
y = np.array(y_data)

print("Final dataset:", X.shape)

np.savez("data/hakim_final.npz", X=X, y=y)
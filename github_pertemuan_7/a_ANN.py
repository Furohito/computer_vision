from data import get_hakim
import numpy as np
import matplotlib.pyplot as plt

# ===== LOAD DATA =====
images, labels = get_hakim()

print("Total Data:", images.shape)

# ===== SHUFFLE (WAJIB) =====
indices = np.arange(len(images))
np.random.shuffle(indices)

images = images[indices]
labels = labels[indices]

# ===== MODEL =====
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.001
epochs = 100

label_names = [
    "circle", "diamond", "ellipse_h", "ellipse_v",
    "parallelogram", "rectangle_h", "rectangle_v",
    "square", "trapezium", "triangle"
]

# ===== TRAINING (SEMUA DATA) =====
for epoch in range(epochs):
    nr_correct = 0

    for img, l in zip(images, labels):
        img = img.reshape(784, 1)
        l = l.reshape(10, 1)

        # FORWARD
        h = 1 / (1 + np.exp(-(b_i_h + w_i_h @ img)))
        o = 1 / (1 + np.exp(-(b_h_o + w_h_o @ h)))

        nr_correct += int(np.argmax(o) == np.argmax(l))

        # BACKPROP
        delta_o = o - l

        w_h_o -= learn_rate * (delta_o @ h.T)
        b_h_o -= learn_rate * delta_o

        delta_h = (w_h_o.T @ delta_o) * (h * (1 - h))

        w_i_h -= learn_rate * (delta_h @ img.T)
        b_i_h -= learn_rate * delta_h

    acc = (nr_correct / len(images)) * 100
    print(f"Epoch {epoch+1} | Accuracy: {round(acc, 2)}%")

# ===== TESTING (INSAMPLE) =====
while True:
    index = int(input(f"Index (0 - {len(images)-1}): "))

    img = images[index].reshape(28, 28)
    plt.imshow(img, cmap="Greys")

    img_vec = images[index].reshape(784, 1)

    h = 1 / (1 + np.exp(-(b_i_h + w_i_h @ img_vec)))
    o = 1 / (1 + np.exp(-(b_h_o + w_h_o @ h)))

    pred = np.argmax(o)

    plt.title(f"Pred: {label_names[pred]}")
    plt.show()
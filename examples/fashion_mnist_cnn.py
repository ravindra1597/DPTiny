import time
import numpy as np

from dptiny import (
    Variable,
    get_fashion_mnist,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_gpu,
    to_cpu,
    use_gpu,
    xp, 
)
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES  # importing the CLASSES tuple
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")

# Setting random seed.
xp.random.seed(42)

print("Loading Fashion-MNIST dataset...")
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)
print("Dataset loaded successfully.")

# Standardizing using the training-set mean and standard deviation.
train_mean = X_train.mean()
train_std = X_train.std()

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

batch_size = 64
max_epoch = 10
data_loader = DataLoader((X_train, y_train), batch_size)
test_loader = DataLoader((X_test, y_test), batch_size, shuffle=False)

optimizer = Adam(model, lr=0.001)

# Training Section
start_time = time.time()
print("Starting training...") # For my convenience in the terminal when I run this code. 
for epoch in range(max_epoch):
    sum_loss = 0.0
    train_correct = 0
    train_count = 0

    model.train()

    for batch_index,(x, t) in enumerate(data_loader):
        x = Variable(x)

        y = model(x)
        loss = softmax_cross_entropy(y, t)

        # Training predictions for accuracy
        pred = y.data.argmax(axis=1)
        train_correct += int((pred == t).sum())

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        train_count += len(t)
        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{max_epoch}, "
                f"Batch {batch_index + 1}/{len(data_loader)}"
            )

    train_loss = sum_loss / train_count
    train_acc = train_correct / train_count

    # Test accuracy for this epoch
    test_correct = 0
    test_count = 0

    with test_mode(), no_grad():
        for x, t in test_loader:
            y = model(Variable(x))
            pred = y.data.argmax(axis=1)

            test_correct += int((pred == t).sum())
            test_count += len(t)

    test_acc = test_correct / test_count

    print(
        f"Epoch {epoch + 1}/{max_epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Acc: {test_acc:.4f} | "
        f"Time: {time.time() - start_time:.2f}s"
    )

# Testing Section
# Final evaluation: collect all predictions and true labels.
all_predictions = []
all_labels = []

with test_mode(), no_grad():
    for x, t in test_loader:
        y = model(Variable(x))
        pred = y.data.argmax(axis=1)

        all_predictions.append(pred)
        all_labels.append(t)

# Combining all batches into single arrays.
all_predictions = xp.concatenate(all_predictions)
all_labels = xp.concatenate(all_labels)

# Converting to CPU/NumPy arrays in case training was done on GPU.
all_predictions = to_cpu(all_predictions)
all_labels = to_cpu(all_labels)

# Building the 10 x 10 confusion matrix.
confusion_matrix = np.zeros((10, 10), dtype=np.int32)

for true_label, predicted_label in zip(all_labels, all_predictions):
    confusion_matrix[int(true_label), int(predicted_label)] += 1

print("\nConfusion Matrix:")
print(confusion_matrix)

# Printing Per class - Accuracy 
print("\nPer-Class Accuracy:")
for class_index, class_name in enumerate(CLASSES):
    correct = confusion_matrix[class_index, class_index]
    total = confusion_matrix[class_index].sum()

    class_accuracy = correct / total if total > 0 else 0.0

    print(
        f"{class_index}: {class_name:12s} "
        f"Accuracy: {class_accuracy:.4f}"
    )

# Finding the pair of different classes confused most often.
confusion_copy = confusion_matrix.copy()

# Ignoring correct predictions on the diagonal.
np.fill_diagonal(confusion_copy, 0)

# Finding the largest off-diagonal entry.
most_confused_index = np.unravel_index(
    np.argmax(confusion_copy),
    confusion_copy.shape
)

true_class, predicted_class = most_confused_index
confusion_count = confusion_copy[true_class, predicted_class]

print("\nMost Confused Pair:")
print(
    f"{CLASSES[true_class]} -> {CLASSES[predicted_class]} "
    f"({confusion_count} times)"
)

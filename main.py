import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


data_dir = 'data'  
target_size = (128, 128) 
epochs = 25

def preprocess_image(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    new_image = np.zeros((target_h, target_w, 3), dtype=resized_image.dtype)
    start_x = (target_w - new_w) // 2
    start_y = (target_h - new_h) // 2
    new_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized_image
    ycrcb = cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return equalized_image


images, labels = [], []
class_names = sorted(os.listdir(data_dir))


for idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    for file in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file)
        image = cv2.imread(file_path)
        if image is not None:
            processed_image = preprocess_image(image, target_size)
            images.append(processed_image)
            labels.append(idx)


images = np.array(images, dtype='float32') / 255.0
labels = to_categorical(np.array(labels), num_classes=len(class_names))


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[1], target_size[0], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")


model.save('invasive_species_classifier.h5')


y_true = np.argmax(y_test, axis=1)

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)



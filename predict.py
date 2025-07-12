import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("invasive_species_classifier.h5")

# Define class labels in the same order as your training data
class_names = ['english_ivy', 'honeysuckle', 'mile_a_minute', 'porcelain_berry']

# Image path (update this to test a different image)
image_path = r"C:\Users\Test\Desktop\sampleimage"

# Preprocess function
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    new_image = np.zeros((target_size[1], target_size[0], 3), dtype=resized_image.dtype)
    start_x = (target_size[0] - new_w) // 2
    start_y = (target_size[1] - new_h) // 2
    new_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized_image

    ycrcb = cv2.cvtColor(new_image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    normalized = equalized_image.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=0)  # shape = (1, 128, 128, 3)

# Predict
image = preprocess_image(image_path)
pred = model.predict(image)
predicted_index = np.argmax(pred)
confidence = pred[0][predicted_index]

print(f"Predicted class: {class_names[predicted_index]}")
print(f"Confidence: {confidence:.2%}")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {pred[0][i]:.2%}")


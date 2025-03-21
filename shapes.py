import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Global variables to store the trained model and scaler
model = None
scaler = None
labels = []
features = []

def extract_hu_moments(image_path):
    """Extracts Hu Moments from an image."""
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        return None
    blurred = cv2.GaussianBlur(original_image, (5, 5), 0)
    thresh_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh_adaptive.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Consider the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments)
    # Log transform to handle the large range of Hu moments
    return -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

def train_model(dataset_path):
    """Trains a K-Nearest Neighbors model using Hu Moments as features."""
    global model, scaler, labels, features
    labels = []
    features = []
    subfolders = ['squares', 'triangles', 'circles']

    for shape_type in subfolders:
        shape_folder_path = os.path.join(dataset_path, shape_type)
        if not os.path.isdir(shape_folder_path):
            print(f"Warning: Subfolder '{shape_type}' not found in {dataset_path}")
            continue

        image_files = [f for f in os.listdir(shape_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        print(f"Extracting features from '{shape_type}' folder...")

        for image_file in image_files:
            image_path = os.path.join(shape_folder_path, image_file)
            hu_moments = extract_hu_moments(image_path)
            if hu_moments is not None:
                features.append(hu_moments.flatten())
                labels.append(shape_type[:-1])  # 'squares' -> 'square'

    features = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = KNeighborsClassifier(n_neighbors=5)  # You can experiment with the number of neighbors
    model.fit(scaled_features, labels)
    print("\nModel training complete.")

def predict_shape(image_path):
    """Predicts the shape in an image using the trained model."""
    if model is None or scaler is None:
        print("Error: Model not trained yet.")
        return "unknown"

    hu_moments = extract_hu_moments(image_path)
    if hu_moments is None:
        return "unknown"

    scaled_hu_moments = scaler.transform(hu_moments.reshape(1, -1))
    predicted_shape = model.predict(scaled_hu_moments)[0]
    return predicted_shape

def analyze_shapes_dataset(dataset_path):
    """
    Analyzes a dataset of shapes using the trained model and displays overall accuracy.

    Args:
        dataset_path (str): The path to the 'shapes' folder.
    """
    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset folder not found at {dataset_path}")
        return

    train_model(dataset_path)  # Train the model before analyzing

    subfolders = ['squares', 'triangles', 'circles']
    results = {'squares': {'correct': 0, 'total': 0},
               'triangles': {'correct': 0, 'total': 0},
               'circles': {'correct': 0, 'total': 0}}

    total_correct = 0
    total_images = 0

    for shape_type in subfolders:
        shape_folder_path = os.path.join(dataset_path, shape_type)
        if not os.path.isdir(shape_folder_path):
            print(f"Warning: Subfolder '{shape_type}' not found in {dataset_path}")
            continue

        image_files = [f for f in os.listdir(shape_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        print(f"\nAnalyzing images in '{shape_type}' folder...")

        for image_file in image_files:
            image_path = os.path.join(shape_folder_path, image_file)
            predicted_shape = predict_shape(image_path)

            results[shape_type]['total'] += 1
            total_images += 1
            if predicted_shape == shape_type[:-1]:
                results[shape_type]['correct'] += 1
                total_correct += 1

            # Optional: Display the image and detected shape for visual inspection
            # original_image = cv2.imread(image_path)
            # if original_image is not None:
            #     cv2.putText(original_image, f"Predicted: {predicted_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #     cv2.imshow(f"Image from {shape_type}", original_image)
            #     cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Print the evaluation results
    print("\n--- Evaluation Results ---")
    for shape_type, counts in results.items():
        accuracy = (counts['correct'] / counts['total']) * 100 if counts['total'] > 0 else 0
        print(f"Accuracy for {shape_type}: {accuracy:.2f}% ({counts['correct']} correct out of {counts['total']})")

    # Calculate and print overall accuracy
    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({total_correct} correct out of {total_images})")

# Example usage:
dataset_path = 'shapes'
analyze_shapes_dataset(dataset_path)
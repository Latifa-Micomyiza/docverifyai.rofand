
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import json

# Load YOLOv8 model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Required ID features & confidence thresholds
required_classes = {
    "ID Number": 0.87,
    "Rwandan Flag": 0.87,
    "Coat of Arms": 0.87,
    "Text Area": 0.85,
}

# Preprocessing function for improving image for text recognition
def preprocess_image(cropped_image):
    """Preprocess image for OCR: convert to grayscale, blur, thresholding, and resize."""
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to enhance contrast
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionally resize the image to a larger size for better recognition
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return resized

# Function to extract text from bounding box
def extract_text(image, box):
    """Extract text from a given bounding box using pytesseract."""
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2]  # Crop the text area

    # Preprocess the cropped image
    preprocessed_image = preprocess_image(cropped_image)

    # Extract text using pytesseract
    text = pytesseract.image_to_string(preprocessed_image, config="--psm 6")
    return text.strip()

# Function to check if the detected features match the Rwandan ID requirements
def is_rwandan_id(results):
    """Check if the image contains all required features for a Rwandan ID."""
    detected_classes = set()
    required_detected_classes = []  # Use a list to store dictionaries

    # Iterate over YOLO detection results
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]  # Get class name
            confidence = float(box.conf)  # Get confidence score

            if class_name in required_classes and confidence >= required_classes[class_name]:
                detected_classes.add(class_name)

            # Collect detected classes for reporting
            if confidence >= 0.70:
                required_detected_classes.append({  # Append to the list
                    "class_name": class_name,
                    "confidence": confidence
                })

    # Validate if all required features are detected
    if detected_classes == set(required_classes.keys()):
        return True, required_detected_classes  # Return valid ID and detected classes info
    else:
        return False, required_detected_classes  # Return invalid ID and detected classes info

# Function to process uploaded image and extract necessary information
def process_image(file_path):
    """Process the uploaded image to detect features and extract text."""
    # Read the image directly from the file path
    image = cv2.imread(file_path)
    if image is None:
        return {"error": "Failed to process image"}

    # Run YOLOv8 inference
    results = model(image)

    # Check if it's a valid Rwandan ID
    id_valid, detected_classes_info = is_rwandan_id(results)

    # Extract text from "Text Area" if present
    extracted_text = None
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            if class_name == "Text Area" and confidence >= required_classes["Text Area"]:
                extracted_text = extract_text(image, box.xyxy[0])

    # Format the result into JSON structure
    output = {
        "id_valid": id_valid,
        "detected_classes_info": detected_classes_info,  # Return all detected classes with confidence
        "extracted_text": extracted_text,
        "message": "Valid Rwandan ID" if id_valid else "Missing required features",
    }

    # Return the result as JSON
    print(output)
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import os

# Load YOLOv8 model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Required ID features & confidence thresholds
required_classes = {
    "Coat of Arms": 0.70,
    "Rwandan Flag": 0.60,  # Lower the threshold
    "ID number": 0.78,
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

    # Debug: Print detected and required classes
    print(f"Detected Classes: {detected_classes}")
    print(f"Required Classes: {set(required_classes.keys())}")

    # Validate if all required features are detected
    validation_passed = all(
        cls in detected_classes for cls in required_classes.keys()
    )

    return validation_passed, required_detected_classes

# Function to transform the extracted text into the desired format
def transform_output(extracted_text_area, extracted_id_number):
    """Transform the extracted text and ID number into the desired format."""
    # Split the text area into lines
    lines = extracted_text_area.split("\n")

    # Define keywords and their corresponding keys
    keywords = {
        "Amazina / Names": {
            "key": "namesOnNationalId",
            "name": "Amazina / Names"
        },
        "ltariki yavutseno / Date of Birth": {
            "key": "dateOfBirth",
            "name": "Itariki yavutseho / Date of Birth"
        },
        "Igitsina/ Sex": {
            "key": "gender",
            "name": "Igitsina / Sex"
        },
        "Aho Yatangiwe / Place of Issue": {
            "key": "placeOfIssue",
            "name": "Aho Yatangiwe / Place of Issue"
        }
    }

    # Initialize the result list
    result = []

    # Iterate through lines and extract key-value pairs
    for i, line in enumerate(lines):
        for keyword, field_info in keywords.items():
            if keyword in line:
                # The value is usually the next line
                if i + 1 < len(lines):
                    value = lines[i + 1].strip()
                    if value:  # Only add non-empty values
                        result.append({
                            "key": field_info["key"],
                            "name": field_info["name"],
                            "value": value
                        })
                break

    # Add the ID number to the result
    if extracted_id_number:
        result.append({
            "key": "nationalId",
            "name": "Indangamuntu / National ID No.",
            "value": extracted_id_number
        })

    # Debug: Print the result
    print("Transformed Result:")
    print(result)

    return result

# Function to process the uploaded image
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

    # Initialize variables to store extracted text
    extracted_text_area = None
    extracted_id_number = None

    # Iterate over YOLO detection results
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)

            # Extract text from "Text Area"
            if class_name == "Text Area" and confidence >= required_classes["Text Area"]:
                extracted_text_area = extract_text(image, box.xyxy[0])
                print("Extracted Text Area:")
                print(extracted_text_area)  # Debug: Print the extracted text

            # Extract text from "ID number"
            if class_name == "ID number" and confidence >= required_classes["ID number"]:
                extracted_id_number = extract_text(image, box.xyxy[0])

    # Format the result into JSON structure
    output = {
        "id_valid": id_valid,
        "detected_classes_info": detected_classes_info,
        "extracted_text_area": extracted_text_area,  # Raw text from Text Area
        "extracted_id_number": extracted_id_number,  # Raw text from ID Number
        "message": "ID successfully authenticated as Rwandan" if id_valid else "Missing required features",
    }

    # Print the raw output (optional)
    print("Raw Output:")
    print(output)

    # Transform the output into the desired format
    if extracted_text_area and extracted_id_number:
        transformed_output = transform_output(extracted_text_area, extracted_id_number)
        print("Transformed Output:")
        print(transformed_output)
    else:
        print("Error: Missing extracted text or ID number.")
        return output

    os.remove(file_path)
    print(f"File {file_path} deleted successfully.")



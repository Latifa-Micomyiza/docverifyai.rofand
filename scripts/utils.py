import cv2
import pytesseract

required_classes = {
    "Hologram": 0.87,
    "ID Number": 0.87,
    "Rwandan Flag": 0.87,
    "Coat of Arms": 0.87,
}

def extract_text(image, box):
    """ Extract text from a detected bounding box """
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2]
    text = pytesseract.image_to_string(cropped_image, config="--psm 6")
    return text.strip()

def is_rwandan_id(results):
    """ Verify if all required elements are detected """
    detected_classes = set()
    for result in results:
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf)
            if class_name in required_classes and confidence >= required_classes[class_name]:
                detected_classes.add(class_name)
    return detected_classes == set(required_classes.keys())

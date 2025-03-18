import cv2
from ultralytics import YOLO
from scripts.utils import extract_text, is_rwandan_id

model = YOLO("runs/detect/train/weights/best.pt")

def process_live_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        is_valid = is_rwandan_id(results)

        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                if class_name == "Text Area" and confidence > 0.90:
                    text = extract_text(frame, box.xyxy[0])
                    print(f"Extracted Text: {text}")

        output_frame = results[0].plot()
        cv2.imshow("Live Camera", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

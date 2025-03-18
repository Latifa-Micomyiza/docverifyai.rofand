from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Load an input image
input_image_path = "inference/input_images/idtest.jpg"
image = cv2.imread(input_image_path)

# Run inference
results = model(image)

# Visualize and save the results
for result in results:
    # Plot the results on the image
    output_image = result.plot()

    # Save the output image
    output_image_path = "inference/output_images/output_image.png"
    cv2.imwrite(output_image_path, output_image)

print(f"Inference complete! Results saved to {output_image_path}")

from ultralytics import YOLO
import cv2
import cvzone
import math

VID_001 = 'sample-files/JapanPPE.mp4'
WEIGHTS = 'runs/train/yolov8x.pt_ppe_100_epochs/weights/best.pt'
CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
CLASSES_TO_DETECT = [0, 2, 4, 5, 7]

cap = cv2.VideoCapture(VID_001)

# Define output video file name and codec
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# Get video frame size from the first frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

model = YOLO(WEIGHTS)

def calculate_enclosed_percentage(box1, box2):
    """
    Calculate the percentage of box1 that is enclosed by box2.

    Parameters:
        box1 (tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2),
                      where (x1, y1) are the coordinates of the top-left corner,
                      and (x2, y2) are the coordinates of the bottom-right corner.
        box2 (tuple): Coordinates of the second bounding box in the same format as box1.

    Returns:
        float: Percentage of box1 that is enclosed by box2, a value between 0 and 1.
    """
    # Extract coordinates of the bounding boxes
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calculate the area of box1
    area_box1 = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)

    # Calculate the area of intersection between box1 and box2
    x1_intersection = max(x1_box1, x1_box2)
    y1_intersection = max(y1_box1, y1_box2)
    x2_intersection = min(x2_box1, x2_box2)
    y2_intersection = min(y2_box1, y2_box2)
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)

    # Calculate the percentage of box1 enclosed by box2
    enclosed_percentage = intersection_area / area_box1
    
    # Ensure the percentage is within [0, 1]
    enclosed_percentage = max(0, min(1, enclosed_percentage))
    
    return enclosed_percentage


while True:
    success, img = cap.read()

    if not success:
        print("Error reading video frame. Exiting...")
        break

    # Perform YOLO object detection
    results = model(img, stream=True, classes=CLASSES_TO_DETECT, conf=0.5)

    # Iterate over detected objects and draw bounding boxes
    for data in results:
        boxes = data.boxes
        for box in boxes:
            # Extract class label
            cls = int(box.cls[0])
            class_label = CLASSES[cls]

            if class_label == 'Person':
                # Extract bounding box coordinates
                xmin, ymin, xmax, ymax = box.xyxy[0]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                # Extract confidence score
                confidence = float(box.conf[0])

                color = (0, 255, 0)

                hardhat_present = False
                safety_vest_present = False
                for obj_box in data.boxes:
                    obj_class_label = CLASSES[int(obj_box.cls[0])]
                    if obj_class_label in ['Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'machinery', 'vehicle']:
                        continue

                    xmin_obj, ymin_obj, xmax_obj, ymax_obj = obj_box.xyxy[0]
                    xmin_obj, ymin_obj, xmax_obj, ymax_obj = int(xmin_obj), int(ymin_obj), int(xmax_obj), int(ymax_obj)

                    enclosed_fraction = calculate_enclosed_percentage((xmin_obj, ymin_obj, xmax_obj, ymax_obj), (xmin, ymin, xmax, ymax))

                    if enclosed_fraction > 0.9:
                        if obj_class_label == 'Safety Vest':
                            safety_vest_present = True
                        if obj_class_label == 'Hardhat':
                            hardhat_present = True

                if safety_vest_present and hardhat_present:
                    color = (0, 255, 0)  # Green when both safety vest and hardhat are detected
                    class_label = "Hardhat + Safety Vest"
                elif safety_vest_present:
                    color = (0, 255, 255)  # Yellow when only safety vest is detected
                    class_label = "Safety Vest Only"
                elif hardhat_present:
                    color = (0, 255, 255)  # Yellow when only hardhat is detected
                    class_label = "Hardhat Only"
                else:
                    color = (0, 0, 255)  # Red when neither safety vest nor hardhat is detected
                    class_label = "No equipment"

                # Draw bounding box and label
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                text_position = (xmin, ymin - 5)  # Adjust based on font size
                cv2.putText(img, f"{class_label}: {confidence:.2f}", text_position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

        # Write frame to output video
        out.write(img)

        # Display the frame with bounding boxes
        cv2.imshow('Personal protective equipment detection', img)

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
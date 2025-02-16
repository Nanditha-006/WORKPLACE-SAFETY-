from ultralytics import YOLO
import cv2 # type: ignore

# Load the PPE Detection Model
ppe_model = YOLO('/Users/nandhu/Desktop/project1/detect/train/weights/best.pt')

# Video Capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform PPE Detection
    results = ppe_model(frame)

    # Draw Bounding Boxes and Labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f'{ppe_model.names[cls]}: {conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the Output Frame
    cv2.imshow('PPE Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

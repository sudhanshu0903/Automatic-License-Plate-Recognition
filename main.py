from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from sort import *
from util import get_car, read_license_plate, write_csv

# Initialize results and SORT tracker
results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Load video
cap = cv2.VideoCapture("videos/sample.mp4")

vehicles = [2, 3, 5, 7]  # car, motorbike, bus, truck

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign license plate to vehicle
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop license plate region
            license_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if license_crop.size == 0:
                continue

            # OCR on license plate
            ocr_result = reader.readtext(license_crop)
            plate_text = ocr_result[0][-2] if len(ocr_result) > 0 else None

            if plate_text:
                # Save result
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': plate_text,
                        'bbox_score': score
                    }
                }

                # Draw bounding boxes on main video
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                #  Increased font scale & thickness for better visibility
                cv2.putText(frame, plate_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)

                # Add zoomed-in view of the license plate
                zoom_x, zoom_y = 20, 20
                zoom_w, zoom_h = 300, 120
                plate_zoom = cv2.resize(license_crop, (zoom_w, zoom_h))
                frame[zoom_y:zoom_y + zoom_h, zoom_x:zoom_x + zoom_w] = plate_zoom

                cv2.rectangle(frame, (zoom_x, zoom_y), (zoom_x + zoom_w, zoom_y + zoom_h), (0, 255, 0), 2)
                
                #  Increased size & thickness for the popup text
                cv2.putText(frame, plate_text, (zoom_x + 10, zoom_y + zoom_h + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)

                cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("License Plate Detection", 960, 540)  # width x height
                cv2.imshow("License Plate Detection", frame)

    # Press Q to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save results to CSV
write_csv(results, './test.csv')
print("✅ Processing complete. Results saved to test.csv.")

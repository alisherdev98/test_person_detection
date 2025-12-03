import cv2
from ultralytics import YOLO
import json
import numpy as np
import time


def load_zones():
    with open('restricted_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['zones'], np.int32)


def main():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture('test.mp4')
    zone_array32 = load_zones()
    last_alarm_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        intrusion = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if cv2.pointPolygonTest(zone_array32, (center_x, center_y), False) >= 0:
                        intrusion = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if intrusion:
            last_alarm_time = time.time()
        
        if time.time() - last_alarm_time < 3:
            cv2.putText(frame, 'ALARM!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
        cv2.polylines(frame, [zone_array32], True, (0, 0, 255), 2)
        cv2.imshow('Person Detection', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


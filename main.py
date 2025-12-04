import cv2
from ultralytics import YOLO
import json
import numpy as np
import time
from deep_sort_realtime.deepsort_tracker import DeepSort


def load_zones():
    with open('restricted_zones.json', 'r') as f:
        data = json.load(f)
    return np.array(data['zones'], np.int32)


def main():
    model = YOLO('yolov8m.pt')
    cap = cv2.VideoCapture('test.mp4')

    zone_array32 = load_zones()
    last_alarm_time = 0

    tracker = DeepSort(max_age=300, embedder_gpu=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)

        detections = []
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    detections.append([[x1, y1, x2 - x1, y2 - y1], conf, 0])
        
        tracks = tracker.update_tracks(detections, frame=frame)
        intrusion = False
        
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            
            
            x1, y1, x2, y2 = [int(x) for x in track.to_ltrb()]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            track_id = track.track_id

            rectangle_args = [frame, (x1, y1), (x2, y2), 'color', 2]
            puttext_args = [frame, f'ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 'color', 2]
            
            if cv2.pointPolygonTest(zone_array32, (center_x, center_y), False) >= 0:
                intrusion = True
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            rectangle_args[3] = color
            puttext_args[5] = color

            cv2.rectangle(*rectangle_args)
            cv2.putText(*puttext_args)

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


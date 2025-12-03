import cv2
import json

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def main():
    cap = cv2.VideoCapture('test.mp4')
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("Не удалось открыть видео")
    
    cv2.namedWindow('Zona')
    cv2.setMouseCallback('Zona', mouse_callback)
    
    while True:
        frame_copy = frame.copy()

        if len(points) > 0:
            for i, point in enumerate(points):
                cv2.circle(frame_copy, tuple(point), 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(frame_copy, tuple(points[i-1]), tuple(point), (0, 0, 255), 2)
            
            if len(points) > 2:
                cv2.line(frame_copy, tuple(points[-1]), tuple(points[0]), (0, 0, 255), 2)
        
        cv2.imshow('Zona', frame_copy)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if len(points) > 2:
        with open('restricted_zones.json', 'w') as f:
            json.dump({"zones": points}, f)


if __name__ == '__main__':
    main()


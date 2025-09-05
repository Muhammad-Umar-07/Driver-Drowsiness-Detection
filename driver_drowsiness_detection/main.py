import cv2
import time
from alarm_system import AlarmSystem  
from drowsiness_detector import DrowsinessDetector

def main():
    detector = DrowsinessDetector()
    alarm_system = AlarmSystem()
    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting drowsiness detection system. Press 'q' to quit.")
    
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            processed_frame, drowsiness_detected = detector.process_frame(frame)
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time
            
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(processed_frame, f"Alarm: {'ON' if alarm_system.alarm_active else 'OFF'}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if drowsiness_detected:
                print(f"Drowsiness detected: EAR={detector.last_ear:.2f}, Eye Counter={detector.eye_counter}, Closed={detector.eye_closed}")
                alarm_system.start_alarm()
            else:
                alarm_system.stop_alarm()
            
            cv2.imshow('Drowsiness Detection System', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("System interrupted by user")
    finally:
        alarm_system.stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        print("System shut down successfully")

if __name__ == "__main__":
    main()
import cv2
import time
import pygame
import threading
import numpy as np
import mediapipe as mp
from collections import deque

class AlarmSystem:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.alarm_active = False
        self.alarm_thread = None
        self.create_beep_sound()

    def create_beep_sound(self):
        sample_rate = 44100
        duration = 1000  # 1-second duration
        frequency = 880
        n_samples = int(round(duration * 0.001 * sample_rate))
        buf = np.zeros((n_samples, 2), dtype=np.int16)
        for s in range(n_samples):
            t = float(s) / sample_rate
            buf[s][0] = int(32767 * 0.5 * np.sin(2 * np.pi * frequency * t))
            buf[s][1] = int(32767 * 0.5 * np.sin(2 * np.pi * frequency * t))
        self.alarm_sound = pygame.mixer.Sound(buffer=buf)

    def start_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_thread = threading.Thread(target=self._alarm_loop)
            self.alarm_thread.daemon = True
            self.alarm_thread.start()

    def stop_alarm(self):
        self.alarm_active = False
        pygame.mixer.stop()
        if self.alarm_thread:
            self.alarm_thread.join(timeout=1.0)
            self.alarm_thread = None

    def _alarm_loop(self):
        while self.alarm_active:
            self.alarm_sound.play(loops=-1)  # Loop indefinitely
            time.sleep(0.1)  # Small delay to prevent CPU overload

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # Lowered for better detection at varying distances
            min_tracking_confidence=0.5   # Lowered for better tracking at varying distances
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 380, 373]
        self.HEAD_POSE_INDICES = [1, 199, 33, 263, 61, 291]
        self.EYE_AR_THRESH = 0.35  # Increased to reduce false negatives
        self.EYE_AR_CONSEC_FRAMES = 5  # Reduced to detect shorter eye closures
        self.HEAD_DOWN_THRESH = 15
        self.eye_counter = 0
        self.eye_closed = False
        self.head_down = False
        self.head_pose_buffer = deque(maxlen=15)
        self.last_ear = 0.0
        self.detection_log = []

    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        try:
            eye_points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            D = np.linalg.norm(eye_points[0] - eye_points[3])
            ear = (A + B) / (2.0 * D) if D != 0 else float('inf')
            return ear
        except Exception as e:
            self.detection_log.append(f"EAR calculation error: {str(e)}")
            return float('inf')

    def calculate_head_pose(self, landmarks, frame_shape):
        try:
            image_points = np.array([
                (landmarks[1].x * frame_shape[1], landmarks[1].y * frame_shape[0]),
                (landmarks[199].x * frame_shape[1], landmarks[199].y * frame_shape[0]),
                (landmarks[33].x * frame_shape[1], landmarks[33].y * frame_shape[0]),
                (landmarks[263].x * frame_shape[1], landmarks[263].y * frame_shape[0]),
                (landmarks[61].x * frame_shape[1], landmarks[61].y * frame_shape[0]),
                (landmarks[291].x * frame_shape[1], landmarks[291].y * frame_shape[0])
            ], dtype="double")
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])
            focal_length = frame_shape[1]
            center = (frame_shape[1]/2, frame_shape[0]/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                pitch = euler_angles[0]
                return pitch, 0, 0
        except Exception as e:
            self.detection_log.append(f"Head pose error: {str(e)}")
        return 0, 0, 0

    def process_frame(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.detection_log.append(f"Face mesh error: {str(e)}")
            return frame, False
        
        self.eye_closed = False
        self.head_down = False
        drowsiness_detected = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                left_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.LEFT_EYE_INDICES)
                right_ear = self.calculate_eye_aspect_ratio(face_landmarks.landmark, self.RIGHT_EYE_INDICES)
                ear = (left_ear + right_ear) / 2.0 if left_ear != float('inf') and right_ear != float('inf') else float('inf')
                if ear < self.EYE_AR_THRESH and ear != float('inf'):
                    self.eye_counter += 1
                    if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.eye_closed = True
                        drowsiness_detected = True
                        cv2.putText(frame, "DROWSINESS ALERT: Eyes Closed!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.eye_counter = 0
                pitch, _, _ = self.calculate_head_pose(face_landmarks.landmark, frame.shape)
                if pitch != 0:
                    self.head_pose_buffer.append(pitch)
                if self.head_pose_buffer:
                    avg_pitch = np.mean(self.head_pose_buffer)
                    if avg_pitch < -self.HEAD_DOWN_THRESH:
                        self.head_down = True
                        drowsiness_detected = True
                        cv2.putText(frame, "HEAD DOWN ALERT!", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.last_ear = ear
                cv2.putText(frame, f"EAR: {ear:.2f}" if ear != float('inf') else "EAR: N/A",
                            (10, frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if self.head_pose_buffer:
                    avg_pitch = np.mean(self.head_pose_buffer)
                    cv2.putText(frame, f"Head Pitch: {avg_pitch:.2f}", (10, frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if self.detection_log:
            print(f"Recent errors: {self.detection_log[-1]}")
            self.detection_log = []
        return frame, drowsiness_detected

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
import cv2
import numpy as np
import mediapipe as mp

class FaceMeshAnalyzer:
    """
    Computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose metrics
    from real-time video frames or static images using MediaPipe.
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 6 Facial Landmark points per eye for EAR formula
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Mouth points for MAR formula
        self.MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]

    @staticmethod
    def _euclidean_dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    def calculate_ear(self, eye_pts):
        """
        EAR Formula: (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        v1 = self._euclidean_dist(eye_pts[1], eye_pts[5])
        v2 = self._euclidean_dist(eye_pts[2], eye_pts[4])
        h = self._euclidean_dist(eye_pts[0], eye_pts[3])
        if h == 0:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def calculate_mar(self, mouth_pts):
        """
        MAR Formula: (||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||) / (2 * ||p1 - p5||)
        """
        v1 = self._euclidean_dist(mouth_pts[1], mouth_pts[7])
        v2 = self._euclidean_dist(mouth_pts[2], mouth_pts[6])
        v3 = self._euclidean_dist(mouth_pts[3], mouth_pts[5])
        h = self._euclidean_dist(mouth_pts[0], mouth_pts[4])
        if h == 0:
            return 0.0
        mar = (v1 + v2 + v3) / (2.0 * h)
        return mar

    def process_frame(self, frame_bgr):
        """
        Processes a BGR image frame and returns:
        - avg_ear (float)
        - mar (float)
        - is_eye_closed (bool) [Threshold EAR < 0.21]
        - is_yawning (bool) [Threshold MAR > 0.60]
        - annotated_frame (BGR image with face mesh overlays)
        """
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        annotated = frame_bgr.copy()
        if not results.multi_face_landmarks:
            return 0.30, 0.10, False, False, annotated

        landmarks = results.multi_face_landmarks[0].landmark

        # Get coordinates for left eye, right eye, and mouth
        l_pts = np.array([(landmarks[idx].x * w, landmarks[idx].y * h) for idx in self.LEFT_EYE])
        r_pts = np.array([(landmarks[idx].x * w, landmarks[idx].y * h) for idx in self.RIGHT_EYE])
        m_pts = np.array([(landmarks[idx].x * w, landmarks[idx].y * h) for idx in self.MOUTH])

        left_ear = self.calculate_ear(l_pts)
        right_ear = self.calculate_ear(r_pts)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = self.calculate_mar(m_pts)

        is_eye_closed = avg_ear < 0.21
        is_yawning = mar > 0.60

        # Draw landmarks overlay
        for pt in np.vstack([l_pts, r_pts, m_pts]):
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

        return avg_ear, mar, is_eye_closed, is_yawning, annotated

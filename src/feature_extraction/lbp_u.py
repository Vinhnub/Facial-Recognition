import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBPVideoFeatureExtractor:

    def __init__(
        self,
        img_size=64,
        P=8,
        R=1,
        color_spaces=("gray", "ycrcb"),
        fps=25,
        window_sec=3
    ):

        self.IMG_SIZE = img_size
        self.P = P
        self.R = R
        self.COLOR_SPACES = color_spaces

        self.FPS = fps
        self.WINDOW_SEC = window_sec

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ==========================
    # FACE DETECT
    # ==========================

    def detect_and_crop_face(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (self.IMG_SIZE, self.IMG_SIZE))

        return face

    # ==========================
    # COLOR CONVERT MULTI
    # ==========================

    def convert_color_spaces(self, face):

        channels = []

        for space in self.COLOR_SPACES:

            if space == "gray":

                img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                channels.append(img)

            elif space == "rgb":

                img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                channels.extend(cv2.split(img))

            elif space == "hsv":

                img = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
                channels.extend(cv2.split(img))

            elif space == "ycrcb":

                img = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
                channels.extend(cv2.split(img))

        return channels

    # ==========================
    # LBP FEATURE
    # ==========================

    def extract_lbp(self, gray):

        lbp = local_binary_pattern(
            gray,
            self.P,
            self.R,
            method="default"
        ).astype(np.uint8)

        bits = ((lbp[..., None] >> np.arange(self.P)) & 1).astype(np.uint8)

        shifted = np.roll(bits, -1, axis=2)
        transitions = np.sum(bits != shifted, axis=2)

        uniform_mask = transitions <= 2

        non_uniform_label = self.P*(self.P-1) + 2

        lbp_uniform = lbp.copy()
        lbp_uniform[~uniform_mask] = non_uniform_label

        n_bins = self.P*(self.P-1) + 3

        hist, _ = np.histogram(
            lbp_uniform.ravel(),
            bins=n_bins,
            range=(0, n_bins)
        )

        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        return hist

    # ==========================
    # FRAME FEATURE
    # ==========================

    def extract_frame_feature(self, face):

        channels = self.convert_color_spaces(face)

        feature = []

        for ch in channels:

            hist = self.extract_lbp(ch)

            feature.extend(hist)

        return np.array(feature)

    # ==========================
    # VIDEO → FRAME FEATURES
    # ==========================

    def extract_video_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)

        features = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            face = self.detect_and_crop_face(frame)

            if face is None:
                continue

            feat = self.extract_frame_feature(face)

            features.append(feat)

        cap.release()

        if len(features) == 0:
            return None

        return np.array(features)

    # ==========================
    # TEMPORAL FEATURE
    # ==========================

    def aggregate_feature(self, features):

        window_size = int(self.FPS * self.WINDOW_SEC)

        if len(features) < window_size:
            return None

        return np.mean(features[:window_size], axis=0)

    # ==========================
    # MAIN FUNCTION
    # ==========================

    def extract(self, video_path):

        frame_features = self.extract_video_frames(video_path)

        if frame_features is None:
            return None

        video_feature = self.aggregate_feature(frame_features)

        return video_feature
    

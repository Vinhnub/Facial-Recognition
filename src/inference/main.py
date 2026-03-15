import cv2
import numpy as np
import time
from src.feature_extraction.lbp_u import *
import joblib


# ==========================
# BUTTON CLASS
# ==========================

class Button:

    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text

    def draw(self, frame):

        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            (0, 200, 0),
            -1
        )

        cv2.putText(
            frame,
            self.text,
            (self.x + 15, self.y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    def is_clicked(self, x, y):

        if self.x < x < self.x + self.w and \
           self.y < y < self.y + self.h:

            return True

        return False


# ==========================
# CAMERA APP
# ==========================

class CameraRecorder:

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

        self.record_button = Button(20, 20, 160, 50, "Record 4s")

        self.recording = False
        self.start_time = None

        self.frames = []

    # ==========================

    def mouse_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            if self.record_button.is_clicked(x, y):

                print("Start recording...")

                self.recording = True
                self.start_time = time.time()
                self.frames = []

    # ==========================

    def run(self):

        cv2.namedWindow("Camera")
        cv2.setMouseCallback("Camera", self.mouse_callback)

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            # DRAW BUTTON
            self.record_button.draw(frame)

            # ==========================
            # RECORDING
            # ==========================

            if self.recording:

                elapsed = time.time() - self.start_time

                self.frames.append(frame.copy())

                cv2.putText(
                    frame,
                    f"Recording {elapsed:.1f}s",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                if elapsed >= 4:

                    self.recording = False

                    print("Recording done")
                    print("Frames:", len(self.frames))

                    # SAVE VIDEO
                    self.save_video()

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    # ==========================
    # SAVE VIDEO
    # ==========================

    def save_video(self):

        if len(self.frames) == 0:
            return

        h, w = self.frames[0].shape[:2]

        out = cv2.VideoWriter(
            "video/recorded_video.avi",
            cv2.VideoWriter_fourcc(*'XVID'),
            25,
            (w, h)
        )

        for f in self.frames:
            out.write(f)

        out.release()

        print("Video saved: video/recorded_video.avi")


# ==========================
# MAIN
# ==========================

app = CameraRecorder()
app.run()

model = joblib.load("face_antispoof_model.pkl")

extractor = LBPVideoFeatureExtractor(color_spaces=["hsv", "ycrcb"])
feature = extractor.extract("video/recorded_video.avi")

y_score = model.predict_proba(feature)[:, 1]
y_pred = model.predict(feature)

print(y_score)
print(y_pred)
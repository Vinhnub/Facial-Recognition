import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import numpy as np
import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL SPOOF =================
class STDN_Plus_Model(nn.Module):
    def __init__(self):
        super(STDN_Plus_Model, self).__init__()

        base_resnet = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(base_resnet.children())[:-2])

        self.decoder_live = nn.Sequential(
            nn.ConvTranspose2d(512,128,4,2,1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,2,1), nn.Sigmoid()
        )

        self.decoder_trace = nn.Sequential(
            nn.ConvTranspose2d(512,128,4,2,1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(16,3,4,2,1), nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,256), nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        features = self.encoder(x)
        _, _ = self.decoder_live(features), self.decoder_trace(features)
        cls_out = self.classifier(features)
        return cls_out


# ================= LOAD MODELS =================
# Spoof model
spoof_model = STDN_Plus_Model().to(DEVICE)
spoof_model.load_state_dict(torch.load("src/train/stdn_plus_pami.pth", map_location=DEVICE))
spoof_model.eval()

# Face recognition model
face_model = joblib.load(r"E:\PythonFile\Project\Facial-Recognition\src\face_recognize\models/knn_model.pkl")
label_map = joblib.load(r"E:\PythonFile\Project\Facial-Recognition\src\face_recognize\models/label_map.pkl")

# Face detector
face_cascade = cv2.CascadeClassifier(
    r"E:\PythonFile\Project\Facial-Recognition\src\face_recognize\haarcascade_frontalface_default.xml"
)

print("All models loaded!")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])


# ================= LBP =================
def lbp_feature(img):
    h, w = img.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):
            center = img[i, j]
            binary = [
                img[i-1,j-1] > center,
                img[i-1,j] > center,
                img[i-1,j+1] > center,
                img[i,j+1] > center,
                img[i+1,j+1] > center,
                img[i+1,j] > center,
                img[i+1,j-1] > center,
                img[i,j-1] > center
            ]
            value = sum([b << k for k, b in enumerate(binary)])
            lbp[i-1,j-1] = value

    hist,_ = np.histogram(lbp.ravel(),256,[0,256])
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


# ================= SPOOF =================
def predict_spoof(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_out = spoof_model(img_tensor)
        pred = torch.argmax(cls_out, dim=1).item()

    return "REAL" if pred == 1 else "FAKE"


# ================= FACE RECOGNITION =================
def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    results = []

    for (x,y,w,h) in faces:
        pad = int(0.2 * w)

        y1 = max(0, y-pad)
        y2 = min(gray.shape[0], y+h+pad)
        x1 = max(0, x-pad)
        x2 = min(gray.shape[1], x+w+pad)

        face = gray[y1:y2, x1:x2]
        face = cv2.resize(face,(80,70))
        face = cv2.equalizeHist(face)

        feature = lbp_feature(face).reshape(1,-1)

        probs = face_model.predict_proba(feature)
        confidence = np.max(probs)
        pred = np.argmax(probs)

        if confidence < 0.5:
            name = "Unknown"
        else:
            name = label_map[pred]

        results.append((x,y,w,h,name,confidence))

    return results


# ================= CAMERA =================
cap = cv2.VideoCapture(0)

label = ""

print("SPACE: Capture | ESC: Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    cv2.putText(display, "SPACE: Capture | ESC: Exit",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # hiển thị kết quả
    if label != "":
        cv2.putText(display, label,
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,255,0), 3)

    cv2.imshow("Anti-Spoof + FaceID", display)

    key = cv2.waitKey(1)

    if key == 27:
        break

    elif key == 32:  # SPACE
        spoof = predict_spoof(frame)

        if spoof == "FAKE":
            label = "FAKE"
            print("FAKE detected")
            faces = recognize_face(frame)
            texts = []
            for (_,_,_,_,name,conf) in faces:
                    texts.append(f"{name} ({conf:.2f})")

            label = "FAKE - " + ", ".join(texts)

        else:
            faces = recognize_face(frame)

            if len(faces) == 0:
                label = "REAL - No face"
            else:
                texts = []
                for (_,_,_,_,name,conf) in faces:
                    texts.append(f"{name} ({conf:.2f})")

                label = "REAL - " + ", ".join(texts)

            print(label)

cap.release()
cv2.destroyAllWindows()
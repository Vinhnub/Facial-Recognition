import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# STDN+ MODEL (ANTI SPOOF)
# =============================

class STDN_Plus_Model(nn.Module):
    def __init__(self):
        super(STDN_Plus_Model, self).__init__()

        base_resnet = models.resnet18(pretrained=False)
        self.encoder = nn.Sequential(*list(base_resnet.children())[:-2])

        self.decoder_live = nn.Sequential(
            nn.ConvTranspose2d(512,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,4,2,1), nn.Sigmoid()
        )

        self.decoder_trace = nn.Sequential(
            nn.ConvTranspose2d(512,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
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
        f = self.encoder(x)
        return self.decoder_live(f), self.decoder_trace(f), self.classifier(f)


stdn_model = STDN_Plus_Model().to(DEVICE)
stdn_model.load_state_dict(torch.load("src/train/stdn_plus_pami.pth", map_location=DEVICE))
stdn_model.eval()

# =============================
# SVM MODEL (FACE RECOGNITION)
# =============================

svm_model = joblib.load("models/best_svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
kpca = joblib.load("models/kpca.pkl")
label_map = joblib.load("models/label_map.pkl")

# =============================
# TRANSFORM STDN+
# =============================

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# =============================
# FEATURE (SVM)
# =============================

def lbp_feature(img):
    h, w = img.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):
            c = img[i,j]
            binary = [
                img[i-1,j-1]>c, img[i-1,j]>c, img[i-1,j+1]>c,
                img[i,j+1]>c,
                img[i+1,j+1]>c, img[i+1,j]>c, img[i+1,j-1]>c,
                img[i,j-1]>c
            ]
            value = sum([b << k for k,b in enumerate(binary)])
            lbp[i-1,j-1] = value

    hist,_ = np.histogram(lbp.ravel(),256,[0,256])
    return hist


def gabor_feature(img):
    kernels = []
    for theta in np.arange(0, np.pi, np.pi/4):
        k = cv2.getGaborKernel((9,9),1.0,theta,np.pi/2,0.5,0)
        kernels.append(k)

    feats = []
    for k in kernels:
        f = cv2.filter2D(img, cv2.CV_32F, k)
        feats.append(f.flatten())

    return np.hstack(feats)


def extract_feature(face):
    return np.hstack([gabor_feature(face), lbp_feature(face)])


# =============================
# FULL PIPELINE
# =============================

def predict_full(image_path):

    # ===== STEP 1: ANTI SPOOF =====
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, _, cls_out = stdn_model(img_tensor)
        pred = torch.argmax(cls_out, dim=1).item()

    if pred == 0:
        print("FAKE ❌")
        return

    print("REAL ✅")

    # ===== STEP 2: FACE RECOGNITION =====
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if len(faces) == 0:
        print("No face detected")
        return

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(80,70))
        face = cv2.equalizeHist(face)

        feature = extract_feature(face).reshape(1,-1)
        feature = scaler.transform(feature)
        feature = kpca.transform(feature)

        pred = svm_model.predict(feature)
        name = label_map[int(pred[0])]

        print("IDENTITY:", name)


# =============================
# TEST
# =============================

predict_full(r"E:\PythonFile\Project\Facial-Recognition\data\test\z7623414547355_787c4c547fc530947ade927cbb1d7125.jpg")
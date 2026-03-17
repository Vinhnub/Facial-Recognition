import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================
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
        I_live, S_trace = self.decoder_live(features), self.decoder_trace(features)
        cls_out = self.classifier(features)
        return I_live, S_trace, cls_out


# ================= LOAD MODEL =================
model = STDN_Plus_Model().to(DEVICE)
model.load_state_dict(torch.load("src/train/stdn_plus_pami.pth", map_location=DEVICE))
model.eval()

print("Model loaded successfully")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])


# ================= PREDICT =================
def predict_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, _, cls_out = model(img_tensor)
        pred = torch.argmax(cls_out, dim=1).item()

    return "REAL" if pred == 1 else "FAKE"


# ================= CAMERA =================
save_dir = "captured"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

print("SPACE: chụp | ESC: thoát")

count = 0
label = ""  # lưu kết quả gần nhất

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # hiển thị hướng dẫn
    cv2.putText(display_frame, "SPACE: Capture | ESC: Exit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2)

    # hiển thị kết quả nếu có
    if label != "":
        color = (0,255,0) if label == "REAL" else (0,0,255)
        cv2.putText(display_frame, f"{label}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, color, 3)

    cv2.imshow("Camera Anti-Spoofing", display_frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

    elif key == 32:  # SPACE
        filename = os.path.join(save_dir, f"capture_{count}.jpg")
        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

        # predict và lưu label
        label = predict_image(frame)
        print("Prediction:", label)

        count += 1

cap.release()
cv2.destroyAllWindows()
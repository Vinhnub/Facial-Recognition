import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
        I_live = self.decoder_live(features)
        S_trace = self.decoder_trace(features)
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


# ================= PREDICT + VISUALIZE =================
def predict_and_show(image_path):

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth_map, trace_map, cls_out = model(img_tensor)
        pred = torch.argmax(cls_out, dim=1).item()

    label = "REAL" if pred == 1 else "FAKE"

    # convert tensor → numpy
    img_np = img_tensor.squeeze().cpu().permute(1,2,0).numpy()
    depth_np = depth_map.squeeze().cpu().numpy()
    trace_np = trace_map.squeeze().cpu().permute(1,2,0).numpy()

    # normalize trace for visualization
    trace_np = (trace_np - trace_np.min()) / (trace_np.max() - trace_np.min())

    # ================= PLOT =================
    # plt.figure(figsize=(12,4))

    # plt.subplot(1,3,1)
    # plt.imshow(img_np)
    # plt.title("Input Image")
    # plt.axis("off")

    # plt.subplot(1,3,2)
    # plt.imshow(depth_np, cmap="jet")
    # plt.title("Predicted Depth")
    # plt.axis("off")

    # plt.subplot(1,3,3)
    # plt.imshow(trace_np)
    # plt.title("Spoof Trace")
    # plt.axis("off")

    # plt.suptitle(f"Prediction: {label}", fontsize=16)

    # plt.show()
    print(label)

# # ================= TEST =================
predict_and_show(r"E:\PythonFile\Project\Facial-Recognition\data\test\z7623414547355_787c4c547fc530947ade927cbb1d7125.jpg")
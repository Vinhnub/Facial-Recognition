import cv2
import numpy as np
import joblib

# ===== LOAD =====
model = joblib.load("models/best_svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
kpca = joblib.load("models/kpca.pkl")
label_map = joblib.load("models/label_map.pkl")


# ===== FEATURE =====
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
    return hist


def gabor_feature(img):
    kernels = []

    for theta in np.arange(0, np.pi, np.pi/4):
        kernel = cv2.getGaborKernel((9,9),1.0,theta,np.pi/2,0.5,0)
        kernels.append(kernel)

    features = []
    for kernel in kernels:
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        features.append(filtered.flatten())

    return np.hstack(features)


# ===== PREDICT =====
def predict_image(img_path):

    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (80,70))
    img = cv2.equalizeHist(img)

    gabor = gabor_feature(img)
    lbp = lbp_feature(img)

    feature = np.hstack([gabor, lbp])

    # scale + kpca
    feature = scaler.transform([feature])
    feature = kpca.transform(feature)

    pred = model.predict(feature)[0]
    prob = model.predict_proba(feature)[0]

    print("Label:", label_map[pred])
    print("Confidence:", prob)


# ===== RUN =====
predict_image(r"E:\PythonFile\Project\Facial-Recognition\data\test\z7623414547355_787c4c547fc530947ade927cbb1d7125.jpg")
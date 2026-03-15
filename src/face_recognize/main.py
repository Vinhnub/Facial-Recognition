import cv2
import joblib
import numpy as np

USE_LBP = True   # False = Gabor only
                  # True  = Gabor + LBP

# =============================
# LOAD MODEL
# =============================

model = joblib.load("models/best_svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
kpca = joblib.load("models/kpca.pkl")
label_map = joblib.load("models/label_map.pkl")

print("USE_LBP =", USE_LBP)

def lbp_feature(img):

    h, w = img.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):

            center = img[i, j]

            binary = [
                img[i-1, j-1] > center,
                img[i-1, j] > center,
                img[i-1, j+1] > center,
                img[i, j+1] > center,
                img[i+1, j+1] > center,
                img[i+1, j] > center,
                img[i+1, j-1] > center,
                img[i, j-1] > center
            ]

            value = sum([b << k for k, b in enumerate(binary)])

            lbp[i-1, j-1] = value

    hist, _ = np.histogram(lbp.ravel(), 256, [0,256])

    return hist


# =============================
# GABOR FEATURE
# =============================

def gabor_feature(img):

    kernels = []

    for theta in np.arange(0, np.pi, np.pi/4):

        kernel = cv2.getGaborKernel(
            (9,9),
            1.0,
            theta,
            np.pi/2,
            0.5,
            0,
            ktype=cv2.CV_32F
        )

        kernels.append(kernel)

    features = []

    for kernel in kernels:

        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)

        features.append(filtered.flatten())

    return np.hstack(features)


# =============================
# FEATURE PIPELINE
# =============================

def extract_feature(face):

    gabor = gabor_feature(face)

    if USE_LBP:
        lbp = lbp_feature(face)
        feature = np.hstack([gabor, lbp])
    else:
        feature = gabor

    return feature


# =============================
# FACE DETECTOR
# =============================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face,(80,70))

        face = cv2.equalizeHist(face)

        # =============================
        # FEATURE
        # =============================

        feature = extract_feature(face).reshape(1,-1)

        print("Feature shape:", feature.shape)

        # =============================
        # SCALE + KPCA
        # =============================

        feature = scaler.transform(feature)
        feature = kpca.transform(feature)

        # =============================
        # PREDICT
        # =============================

        pred = model.predict(feature)

        name = label_map[int(pred[0])]

        # =============================
        # DRAW
        # =============================

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(
            frame,
            name,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
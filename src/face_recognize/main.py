import cv2
import joblib
import numpy as np

model = joblib.load(r"E:\PythonFile\Project\Facial-Recognition\src\face_recognize\models/lbp_model.pkl")
label_map = joblib.load(r"E:\PythonFile\Project\Facial-Recognition\src\face_recognize\models/label_map.pkl")

# =============================
# LBP FEATURE
# =============================
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


# =============================
# LOAD IMAGE
# =============================
img_path = r"C:\Users\admin\Downloads\dataset\test\z7623391816667_1ae7a8d6802324466546ab4251ea5add.jpg"

img = cv2.imread(img_path)

if img is None:
    print("Cannot read image")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =============================
# FACE DETECT
# =============================
face_cascade = cv2.CascadeClassifier(r"E:\PythonFile\Project\Facial-Recognition\src\face_recognize\haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=6,
    minSize=(50,50)
)

if len(faces) == 0:
    print("No face detected")
    exit()

# =============================
# PREDICT
# =============================
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

    probs = model.predict_proba(feature)
    confidence = np.max(probs)
    pred = np.argmax(probs)

    if confidence < 0.85:
        name = "Unknown"
    else:
        name = label_map[pred]

    print(f"{name} ({confidence:.2f})")

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(
        img,
        f"{name} ({confidence:.2f})",
        (x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
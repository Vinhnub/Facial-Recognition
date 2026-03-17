import cv2
import os
import time

person_name = input("Enter person name: ")

dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)

os.makedirs(person_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier("E:/PythonFile/Project/Facial-Recognition/src/face_recognize/haarcascade_frontalface_default.xml")

if face_cascade.empty():
    print("Error loading Haar Cascade")
    exit()

cap = cv2.VideoCapture(0)

count = 0
MAX_IMAGES = 50

print("\nAuto capturing 50 images...")
print("Press Q to quit\n")

last_capture = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot access camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50,50)
    )

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face,(80,70))

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # hiển thị số ảnh
        cv2.putText(frame,
                    f"Images: {count}/{MAX_IMAGES}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        # chụp mỗi 1s
        if time.time() - last_capture > 1 and count < MAX_IMAGES:

            file_path = os.path.join(person_path, f"{count}.jpg")

            cv2.imwrite(file_path, face)

            print(f"Saved: {file_path}")

            count += 1

            last_capture = time.time()

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if count >= MAX_IMAGES:
        print("\nCaptured 50 images successfully!")
        break


cap.release()
cv2.destroyAllWindows()
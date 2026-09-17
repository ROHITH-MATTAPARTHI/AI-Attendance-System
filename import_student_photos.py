import cv2
import os
import re
import shutil

PHOTO_FOLDER = "students photos"
DATASET_FOLDER = "dataset"

os.makedirs(DATASET_FOLDER, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

processed = 0
failed = 0

for filename in os.listdir(PHOTO_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Get student ID from filename
    student_id = os.path.splitext(filename)[0]
    student_id = re.sub(r"\s*\(\d+\)$", "", student_id)

    filepath = os.path.join(PHOTO_FOLDER, filename)

    image = cv2.imread(filepath)

    if image is None:
        print(f"❌ Cannot read: {filename}")
        failed += 1
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        print(f"⚠️ No face found: {filename}")
        failed += 1
        continue

    # Use the largest detected face
    face = max(faces, key=lambda r: r[2] * r[3])

    x, y, w, h = face

    face_image = gray[y:y+h, x:x+w]

    student_folder = os.path.join(DATASET_FOLDER, student_id)
    os.makedirs(student_folder, exist_ok=True)

    # Save face image
    output_file = os.path.join(
        student_folder,
        f"{student_id}.jpg"
    )

    cv2.imwrite(output_file, face_image)

    print(f"✅ {student_id}")

    processed += 1

print("\n==============================")
print("PHOTO IMPORT COMPLETE")
print("==============================")
print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print("==============================")
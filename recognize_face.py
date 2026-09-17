import cv2
import json
import os

# -----------------------------
# File paths
# -----------------------------
model_path = "models/face_model.yml"
labels_path = "models/labels.json"
cascade_path = "haarcascade_frontalface_default.xml"

# -----------------------------
# Load face recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# -----------------------------
# Load student labels
# -----------------------------
with open(labels_path, "r") as file:
    labels = json.load(file)

# -----------------------------
# Student names
# -----------------------------
student_names = {
    "25B21A4533": "ROHITH MATTAPARTHI"
}

# -----------------------------
# Load face detector
# -----------------------------
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("❌ Could not load face detector.")
    exit()

# -----------------------------
# Open camera
# -----------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened.")
    exit()

print("\n================================")
print("       AI FACE RECOGNITION")
print("================================")
print("Look at the camera.")
print("Press Q to quit.\n")

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read camera.")
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        # Recognize face
        label, confidence = recognizer.predict(face)

        # LBPH: lower confidence = better match
        if confidence < 70:

            student_id = labels[str(label)]
            student_name = student_names.get(
                student_id,
                "Unknown Student"
            )

            text = f"{student_name}"
            id_text = f"ID: {student_id}"
            conf_text = f"Confidence: {confidence:.1f}"

        else:

            text = "UNKNOWN"
            id_text = ""
            conf_text = f"Confidence: {confidence:.1f}"

        # Face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Student name
        cv2.putText(
            frame,
            text,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Student ID
        if id_text:
            cv2.putText(
                frame,
                id_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Confidence
        cv2.putText(
            frame,
            conf_text,
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    # Show camera
    cv2.imshow(
        "AI Attendance - Face Recognition",
        frame
    )

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# -----------------------------
# Close camera
# -----------------------------
camera.release()
cv2.destroyAllWindows()

print("\n✅ Face recognition stopped.")


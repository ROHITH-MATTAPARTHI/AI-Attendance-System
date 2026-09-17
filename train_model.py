import cv2
import os
import json

# -----------------------------
# Paths
# -----------------------------
dataset_path = "dataset"
model_path = "models/face_model.yml"
labels_path = "models/labels.json"

# -----------------------------
# Create LBPH recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}

current_label = 0

print("\n================================")
print("       AI FACE TRAINING")
print("================================\n")

# -----------------------------
# Read student folders
# -----------------------------
for student_id in os.listdir(dataset_path):

    student_folder = os.path.join(dataset_path, student_id)

    if not os.path.isdir(student_folder):
        continue

    print(f"📚 Loading student: {student_id}")

    label_map[current_label] = student_id

    image_count = 0

    # -----------------------------
    # Read student's images
    # -----------------------------
    for filename in os.listdir(student_folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(
            student_folder,
            filename
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"⚠️ Could not read: {image_path}")
            continue

        faces.append(image)
        labels.append(current_label)

        image_count += 1

    print(f"   ✅ {image_count} images loaded")

    current_label += 1

# -----------------------------
# Check dataset
# -----------------------------
if len(faces) == 0:
    print("\n❌ No face images found.")
    exit()

print("\n--------------------------------")
print(f"Total training images: {len(faces)}")
print(f"Total students: {current_label}")
print("--------------------------------\n")

# -----------------------------
# Train model
# -----------------------------
print("🧠 Training face-recognition model...")

recognizer.train(
    faces,
    __import__("numpy").array(labels)
)

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)

recognizer.write(model_path)

# -----------------------------
# Save student labels
# -----------------------------
with open(labels_path, "w") as file:
    json.dump(label_map, file, indent=4)

print("\n================================")
print("✅ TRAINING COMPLETED")
print("================================")
print(f"Model saved: {model_path}")
print(f"Labels saved: {labels_path}")
print(f"Students trained: {current_label}")
print(f"Images trained: {len(faces)}")

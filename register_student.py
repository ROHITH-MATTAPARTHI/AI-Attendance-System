import cv2
import os
import sys
import subprocess


# ============================================================
# GET STUDENT DETAILS
# ============================================================

if len(sys.argv) >= 3:

    student_id = sys.argv[1].strip()
    student_name = sys.argv[2].strip()

else:

    student_id = input("Enter Student ID: ").strip()
    student_name = input("Enter Student Name: ").strip()


# ============================================================
# VALIDATE STUDENT DETAILS
# ============================================================

if not student_id or not student_name:

    print("❌ Student ID and name are required.")
    exit()


# ============================================================
# CREATE DATASET FOLDER
# ============================================================

folder = os.path.join(
    "dataset",
    student_id
)

os.makedirs(
    folder,
    exist_ok=True
)


# ============================================================
# OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Camera could not be opened.")
    exit()


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():

    print("❌ Face detector could not be loaded.")

    camera.release()
    exit()


# ============================================================
# SETTINGS
# ============================================================

TARGET_IMAGES = 50

count = 0

last_face = None

# Minimum distance between saved images
MIN_FACE_DISTANCE = 12

# Minimum face size
MIN_FACE_WIDTH = 120
MIN_FACE_HEIGHT = 120


# ============================================================
# START MESSAGE
# ============================================================

print("\n📸 IMPROVED FACE REGISTRATION")
print("--------------------------------")
print(f"Student ID   : {student_id}")
print(f"Student Name : {student_name}")
print()
print("Look directly at the camera.")
print("Slowly move your head left and right.")
print("Also move slightly up and down.")
print("Try to keep your face clearly visible.")
print()
print("Press Q to cancel.")
print("--------------------------------\n")


# ============================================================
# FACE REGISTRATION
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("❌ Could not read camera.")
        break


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Improve contrast
    # --------------------------------------------------------

    gray_equalized = cv2.equalizeHist(gray)


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_detector.detectMultiScale(
        gray_equalized,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT)
    )


    # --------------------------------------------------------
    # Only continue when exactly ONE face is visible
    # --------------------------------------------------------

    if len(faces) == 1:

        (x, y, w, h) = faces[0]


        # ----------------------------------------------------
        # Face quality check
        # ----------------------------------------------------

        face = gray[
            y:y+h,
            x:x+w
        ]


        # Calculate brightness
        brightness = face.mean()


        # ----------------------------------------------------
        # Reject extremely dark / bright faces
        # ----------------------------------------------------

        if brightness < 45:

            message = "Too dark"

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                message,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        elif brightness > 220:

            message = "Too bright"

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                message,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        else:

            # ------------------------------------------------
            # Check if this frame is different enough
            # ------------------------------------------------

            should_save = True

            if last_face is not None:

                # Resize both images to same size
                current_small = cv2.resize(
                    face,
                    (100, 100)
                )

                previous_small = cv2.resize(
                    last_face,
                    (100, 100)
                )


                # Calculate average pixel difference
                difference = cv2.absdiff(
                    current_small,
                    previous_small
                ).mean()


                if difference < MIN_FACE_DISTANCE:

                    should_save = False


            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            if should_save and count < TARGET_IMAGES:

                count += 1


                filename = os.path.join(
                    folder,
                    f"{count}.jpg"
                )


                # Resize face to standard size
                face_resized = cv2.resize(
                    face,
                    (200, 200)
                )


                cv2.imwrite(
                    filename,
                    face_resized
                )


                last_face = face.copy()


            # ------------------------------------------------
            # Draw successful detection
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Images: {count}/{TARGET_IMAGES}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    elif len(faces) > 1:

        # ----------------------------------------------------
        # More than one face
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Only ONE person should be visible",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    else:

        # ----------------------------------------------------
        # No face detected
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Face not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Move slowly: Left / Right / Up / Down",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Improved Student Face Registration",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(100) & 0xFF


    # Press Q to cancel
    if key == ord("q"):

        print("\n⚠️ Registration cancelled.")
        break


    # Automatically stop after 50 images
    if count >= TARGET_IMAGES:

        break


# ============================================================
# RELEASE CAMERA
# ============================================================

camera.release()

cv2.destroyAllWindows()


# ============================================================
# REGISTRATION RESULT
# ============================================================

print("\n--------------------------------")
print("📸 REGISTRATION COMPLETED")
print("--------------------------------")

print(f"Student ID      : {student_id}")
print(f"Student Name    : {student_name}")
print(f"Images Captured : {count}")


# ============================================================
# AUTOMATIC MODEL TRAINING
# ============================================================

if count >= TARGET_IMAGES:

    print("\n🧠 Training face recognition model...")
    print("--------------------------------")


    result = subprocess.run(
        [
            "venv/bin/python",
            "train_model.py"
        ],
        capture_output=True,
        text=True
    )


    print(result.stdout)


    if result.returncode == 0:

        print("✅ Face recognition model updated successfully.")

    else:

        print("❌ Model training failed.")
        print(result.stderr)

else:

    print(
        "\n⚠️ Only "
        f"{count}/{TARGET_IMAGES} "
        "images were captured."
    )

    print("Model training was skipped.")
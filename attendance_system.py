import cv2
import json
import sqlite3
import os
from datetime import datetime


# ==========================================
# AI ATTENDANCE SYSTEM
# ==========================================

MODEL_PATH = "models/face_model.yml"
LABELS_PATH = "models/labels.json"
STUDENTS_PATH = "students.json"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DATABASE_PATH = "attendance/attendance.db"


# ==========================================
# RECOGNITION SETTINGS
# ==========================================

# Lower LBPH confidence/distance is better.
RECOGNITION_THRESHOLD = 70

# Number of successful recognition frames
# required before attendance is marked.
REQUIRED_CONFIRMATIONS = 5

# Minimum acceptable face size.
MIN_FACE_SIZE = 100


# ==========================================
# CHECK REQUIRED FILES
# ==========================================

for file_path in [
    MODEL_PATH,
    LABELS_PATH,
    STUDENTS_PATH,
    CASCADE_PATH
]:

    if not os.path.exists(file_path):

        print(
            f"❌ Required file not found: {file_path}"
        )

        exit()


# ==========================================
# LOAD FACE MODEL
# ==========================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(
    MODEL_PATH
)


# ==========================================
# LOAD LABELS
# ==========================================

with open(
    LABELS_PATH,
    "r"
) as file:

    labels = json.load(file)


labels = {
    int(k): v
    for k, v in labels.items()
}


# ==========================================
# LOAD STUDENT DATABASE
# ==========================================

with open(
    STUDENTS_PATH,
    "r"
) as file:

    students = json.load(file)


print(
    f"✅ Student database loaded: {len(students)} students"
)


# ==========================================
# DATABASE
# ==========================================

os.makedirs(
    "attendance",
    exist_ok=True
)


connection = sqlite3.connect(
    DATABASE_PATH
)

cursor = connection.cursor()


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        student_name TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        status TEXT NOT NULL,
        UNIQUE(student_id, date)
    )
    """
)


connection.commit()


# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)


if face_detector.empty():

    print("❌ Could not load face detector.")

    connection.close()

    exit()


# ==========================================
# MARK ATTENDANCE
# ==========================================

def mark_attendance(
    student_id,
    student_name
):

    now = datetime.now()

    date = now.strftime(
        "%Y-%m-%d"
    )

    time = now.strftime(
        "%H:%M:%S"
    )


    # --------------------------------------
    # Check whether already marked today
    # --------------------------------------

    cursor.execute(
        """
        SELECT id
        FROM attendance
        WHERE student_id = ?
          AND date = ?
        """,
        (
            student_id,
            date
        )
    )


    existing = cursor.fetchone()


    if existing:

        return False


    # --------------------------------------
    # Insert attendance
    # --------------------------------------

    cursor.execute(
        """
        INSERT INTO attendance
        (
            student_id,
            student_name,
            date,
            time,
            status
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            student_id,
            student_name,
            date,
            time,
            "Present"
        )
    )


    connection.commit()


    # --------------------------------------
    # Display attendance message
    # --------------------------------------

    print(
        "\n================================"
    )

    print(
        "✅ ATTENDANCE MARKED"
    )

    print(
        "================================"
    )

    print(
        f"Student : {student_name}"
    )

    print(
        f"ID      : {student_id}"
    )

    print(
        f"Date    : {date}"
    )

    print(
        f"Time    : {time}"
    )

    print(
        "Status  : Present"
    )

    print(
        "================================\n"
    )


    return True


# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print(
        "❌ Camera could not be opened."
    )

    connection.close()

    exit()


# ==========================================
# START MESSAGE
# ==========================================

print(
    "\n================================"
)

print(
    "       AI ATTENDANCE SYSTEM"
)

print(
    "================================"
)

print(
    "Look at the camera."
)

print(
    "Face verification is required."
)

print(
    "Press Q to quit."
)

print(
    "================================\n"
)


# ==========================================
# SESSION DATA
# ==========================================

session_marked = set()

# Student currently being verified
current_student_id = None

# Consecutive confirmation count
confirmation_count = 0

# Stores latest confidence value
last_confidence = {}


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "❌ Could not read camera."
        )

        break


    # --------------------------------------
    # Mirror camera
    # --------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------
    # Convert to grayscale
    # --------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(
            MIN_FACE_SIZE,
            MIN_FACE_SIZE
        )
    )


    # ======================================
    # NO FACE
    # ======================================

    if len(faces) == 0:


        current_student_id = None
        confirmation_count = 0
        
        cv2.putText(
            frame,
            "Looking for face...",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # ======================================
    # PROCESS FACES
    # ======================================

    for (
        x,
        y,
        w,
        h
    ) in faces:


        # ----------------------------------
        # Crop face
        # ----------------------------------

        face = gray[
            y:y+h,
            x:x+w
        ]


        try:

            # ----------------------------------
            # Face recognition
            # ----------------------------------

            label, confidence = recognizer.predict(
                face
            )


            # ----------------------------------
            # Recognized student
            # ----------------------------------

            if (
                confidence < RECOGNITION_THRESHOLD
                and label in labels
            ):


                student_id = str(
                    labels[label]
                )


                # ----------------------------------
                # Find student
                # ----------------------------------

                student = students.get(
                    student_id
                )


                if student:

                    student_name = student.get(
                        "name",
                        student_id
                    )

                else:

                    student_name = student_id


                # ----------------------------------
                # Store confidence
                # ----------------------------------

                last_confidence[
                    student_id
                ] = confidence


                # ----------------------------------
                # Confirmation counter
                # ----------------------------------

                # ----------------------------------
                # STRICT CONSECUTIVE VERIFICATION
                # ----------------------------------

                if current_student_id == student_id:

                   confirmation_count += 1

                else:

                    # Different student detected.
                    # Start verification again.
                    current_student_id = student_id
                    confirmation_count = 1


                current_count = confirmation_count


                # ==================================
                # ATTENDANCE VERIFICATION
                # ==================================

                if (
                    current_count
                    >= REQUIRED_CONFIRMATIONS
                ):


                    # --------------------------------
                    # Mark only once in this session
                    # --------------------------------

                    if student_id not in session_marked:


                        marked = mark_attendance(
                            student_id,
                            student_name
                        )


                        session_marked.add(
                            student_id
                        )


                    status = "PRESENT"


                    status_text = (
                        "VERIFIED - PRESENT"
                    )


                else:

                    status = "VERIFYING"


                    status_text = (
                        f"VERIFYING "
                        f"{display_count}/"
                        f"{REQUIRED_CONFIRMATIONS}"
                    )


                # ==================================
                # GREEN FACE BOX
                # ==================================

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )


                # ==================================
                # STUDENT NAME
                # ==================================

                cv2.putText(
                    frame,
                    student_name,
                    (x, y - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )


                # ==================================
                # STUDENT ID
                # ==================================

                cv2.putText(
                    frame,
                    f"ID: {student_id}",
                    (x, y - 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )


                # ==================================
                # CONFIDENCE
                # ==================================

                cv2.putText(
                    frame,
                    f"Match: {confidence:.1f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )


                # ==================================
                # VERIFICATION STATUS
                # ==================================

                cv2.putText(
                    frame,
                    status_text,
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )


            # ======================================
            # UNKNOWN FACE
            # ======================================

            else:

                # Reset verification because the face
                # is not recognized.
                current_student_id = None
                confirmation_count = 0

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    "UNKNOWN",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Match: {confidence:.1f}",
                    (x, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )


        except Exception as error:

            print(
                "Recognition error:",
                error
            )


    # ======================================
    # TOP STATUS
    # ======================================

    cv2.putText(
        frame,
        "Face Recognition Active",
        (20, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "AI Attendance System",
        frame
    )


    # ======================================
    # KEYBOARD
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ==========================================
# CLOSE EVERYTHING
# ==========================================

camera.release()

cv2.destroyAllWindows()

connection.close()


print(
    "\n✅ AI Attendance System stopped."
)
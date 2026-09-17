from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import os
import json
import sqlite3
import subprocess
import base64
import re
import secrets
import shutil
import numpy as np
import cv2
from datetime import datetime
from openpyxl import Workbook

app = Flask(__name__)

# ==========================================
# SECRET KEY
# ==========================================

app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(32)

if "FLASK_SECRET_KEY" not in os.environ:
    print("⚠️ FLASK_SECRET_KEY is not set; existing sessions will expire on restart.")


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_PATH = os.path.join(BASE_DIR, "students.json")
DATABASE_PATH = os.path.join(BASE_DIR, "attendance", "attendance.db")
ATTENDANCE_PATH = os.path.dirname(DATABASE_PATH)
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_model.yml")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MAX_IMAGE_BYTES = 5 * 1024 * 1024
FACE_IMAGES_REQUIRED = 20
STUDENT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}")


# ==========================================
# LOGIN DETAILS
# ==========================================

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


# ==========================================
# LOAD STUDENT DATABASE
# ==========================================

if not os.path.exists(STUDENTS_PATH):
    print("❌ students.json not found.")
    exit()

with open(STUDENTS_PATH, "r") as file:
    students = json.load(file)

print(f"✅ Student database loaded: {len(students)} students")

recognizer = None
labels = {}
face_detector = None

def valid_student_id(student_id):
    return bool(STUDENT_ID_PATTERN.fullmatch(student_id))


def decode_camera_image(image_data):
    """Decode a base64 camera frame while bounding the request size."""
    if not isinstance(image_data, str):
        raise ValueError("Image must be a base64 string.")

    encoded = image_data.split(",", 1)[-1]
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3) + 4:
        raise ValueError("Image is too large.")

    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Image is not valid base64 data.") from error


def reload_recognition_assets():
    """Atomically replace recognition assets after a successful training run."""
    global recognizer, labels, face_detector

    if not all(os.path.exists(path) for path in (MODEL_PATH, LABELS_PATH, CASCADE_PATH)):
        return False

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        return False

    try:
        loaded_recognizer = cv2.face.LBPHFaceRecognizer_create()
        loaded_recognizer.read(MODEL_PATH)
        with open(LABELS_PATH, "r") as file:
            loaded_labels = {int(key): value for key, value in json.load(file).items()}
    except (cv2.error, OSError, ValueError, json.JSONDecodeError):
        return False

    recognizer = loaded_recognizer
    labels = loaded_labels
    face_detector = detector
    return True


if not reload_recognition_assets():
    print("⚠️ Face-recognition assets are unavailable or invalid.")


# ==========================================
# LOGIN PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"].strip()
        password = request.form["password"].strip()

        # ADMIN LOGIN
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

            session.clear()
            session["user_type"] = "admin"

            return redirect(url_for("admin_dashboard"))

        # STUDENT LOGIN
        if username in students:

            if password == username:

                session.clear()

                session["user_type"] = "student"
                session["student_id"] = username

                return redirect(url_for("student_dashboard"))

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")


# ==========================================
# ADMIN DASHBOARD
# ==========================================

@app.route("/admin")
def admin_dashboard():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    os.makedirs(ATTENDANCE_PATH, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    total_students = len(students)

    cursor.execute(
        """
        SELECT student_id, student_name, time, status
        FROM attendance
        WHERE date = ?
        ORDER BY time DESC
        """,
        (today,)
    )

    today_attendance = cursor.fetchall()

    present_today = 0

    for record in today_attendance:

        if record["status"].lower() == "present":
            present_today += 1

    absent_today = total_students - present_today

    if absent_today < 0:
        absent_today = 0

    if total_students > 0:

        attendance_percentage = round(
            (present_today / total_students) * 100,
            2
        )

    else:

        attendance_percentage = 0

    connection.close()

    return render_template(
        "admin_dashboard.html",
        total_students=total_students,
        present_today=present_today,
        absent_today=absent_today,
        attendance_percentage=attendance_percentage,
        today_attendance=today_attendance,
        today=today
    )


# ==========================================
# START FACE ATTENDANCE
# ==========================================

@app.route("/mobile-attendance")
def mobile_attendance():
    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    return render_template("mobile_attendance.html")

@app.route("/api/recognize", methods=["POST"])
def recognize_api():

    if session.get("user_type") != "admin":
        return jsonify({
            "message": "Unauthorized"
        }), 401


    if recognizer is None or face_detector is None:
        return jsonify({
            "message": "Face recognition model is not available"
        }), 500


    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({
            "message": "No image received"
        }), 400


    try:

        # ==============================
        # DECODE IMAGE
        # ==============================

        image_bytes = decode_camera_image(data["image"])

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:
            return jsonify({
                "message": "Invalid image"
            }), 400


        # ==============================
        # GRAYSCALE
        # ==============================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ==============================
        # FACE DETECTION
        # ==============================

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100)
        )


        # ==============================
        # NO FACE
        # ==============================

        if len(faces) == 0:

            session["verify_student_id"] = None
            session["verify_count"] = 0

            return jsonify({
                "message": "👤 Face not detected",
                "verification_count": 0,
                "attendance_marked": False,
                "already_verified": False
            })


        # ==============================
        # MULTIPLE FACES
        # ==============================

        if len(faces) > 1:

            session["verify_student_id"] = None
            session["verify_count"] = 0

            return jsonify({
                "message": "⚠️ Only one person should be visible",
                "verification_count": 0,
                "attendance_marked": False,
                "already_verified": False
            })


        # ==============================
        # ONE FACE
        # ==============================

        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (200, 200)
        )


        # ==============================
        # LBPH RECOGNITION
        # ==============================

        label, confidence = recognizer.predict(face)


        # Lower confidence = better match

        if confidence >= 70:

            session["verify_student_id"] = None
            session["verify_count"] = 0

            return jsonify({
                "message": "❓ Face not recognized",
                "verification_count": 0,
                "attendance_marked": False,
                "already_verified": False
            })


        # ==============================
        # GET STUDENT ID
        # ==============================

        student_id = labels.get(label)


        if not student_id or student_id not in students:

            session["verify_student_id"] = None
            session["verify_count"] = 0

            return jsonify({
                "message": "❓ Student not recognized",
                "verification_count": 0,
                "attendance_marked": False,
                "already_verified": False
            })


        student = students[student_id]

        student_name = student.get(
            "name",
            student_id
        )


        # ==============================
        # 5 FRAME VERIFICATION
        # ==============================

        previous_student =session.get("verify_student_id")

        previous_count =session.get("verify_count", 0)


        if previous_student == student_id:

            verify_count = previous_count + 1

        else:

            verify_count = 1


        session["verify_student_id"] = student_id

        session["verify_count"] = verify_count


        # ==============================
        # NOT VERIFIED YET
        # ==============================

        if verify_count < 5:

            return jsonify({

                "message":
                    f"🔍 Verifying {verify_count}/5 — {student_name}",

                "student_name":
                    student_name,

                "student_id":
                    student_id,

                "confidence":
                    round(float(confidence), 2),

                "verification_count":
                    verify_count,

                "attendance_marked":
                    False,

                "already_verified":
                    False

            })


        # ==============================
        # ATTENDANCE DATABASE
        # ==============================

        os.makedirs(ATTENDANCE_PATH, exist_ok=True)


        connection = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = connection.cursor()


        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                student_name TEXT,
                date TEXT,
                time TEXT,
                status TEXT,
                UNIQUE(student_id, date)
            )
        """)


        today = datetime.now().strftime(
            "%Y-%m-%d"
        )

        current_time = datetime.now().strftime(
            "%H:%M:%S"
        )


        # ==============================
        # CHECK TODAY'S ATTENDANCE
        # ==============================

        cursor.execute("""
            SELECT student_id
            FROM attendance
            WHERE student_id = ?
            AND date = ?
        """, (
            student_id,
            today
        ))


        already_marked = cursor.fetchone() is not None


        # ==============================
        # NEW ATTENDANCE
        # ==============================

        if not already_marked:

            cursor.execute("""
                INSERT INTO attendance
                (
                    student_id,
                    student_name,
                    date,
                    time,
                    status
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                student_id,
                student_name,
                today,
                current_time,
                "Present"
            ))


            connection.commit()

            connection.close()


            return jsonify({

                "message":
                    "✅ Attendance marked successfully",

                "student_name":
                    student_name,

                "name":
                    student_name,

                "student_id":
                    student_id,

                "confidence":
                    round(float(confidence), 2),

                "verification_count":
                    5,

                "attendance_marked":
                    True,

                "already_verified":
                    False

            })


        # ==============================
        # ALREADY VERIFIED TODAY
        # ==============================

        connection.close()


        return jsonify({

            "message":
                "⚠️ Attendance already verified today",

            "student_name":
                student_name,

            "name":
                student_name,

            "student_id":
                student_id,

            "confidence":
                round(float(confidence), 2),

            "verification_count":
                5,

            "attendance_marked":
                False,

            "already_verified":
                True

        })


    except ValueError as error:

        return jsonify({
            "message": str(error),
            "attendance_marked": False,
            "already_verified": False
        }), 400

    except Exception as e:

        print(
            "Recognition API error:",
            e
        )


        return jsonify({

            "message":
                "Recognition error",

            "attendance_marked":
                False,

            "already_verified":
                False

        }), 500

@app.route("/start-attendance")
def start_attendance():
    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    return redirect(url_for("mobile_attendance"))

# ==========================================
# ATTENDANCE REPORTS
# ==========================================

@app.route("/reports")
def attendance_reports():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    # Get selected date, or use today's date
    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT student_id,
               student_name,
               MIN(time) AS time,
               status
        FROM attendance
        WHERE date = ?
        GROUP BY student_id
        ORDER BY MIN(time) ASC
        """,
        (selected_date,)
    )

    attendance_records = cursor.fetchall()

    present_ids = set()
    present_records = []

    for record in attendance_records:

        if record["status"].lower() == "present":

            present_ids.add(
                str(record["student_id"])
            )

            present_records.append(record)

    absent_students = []

    for student_id, student in students.items():

        if str(student_id) not in present_ids:

            absent_students.append({
                "student_id": student_id,
                "student_name": student.get(
                    "name",
                    student_id
                ),
                "status": "Absent"
            })

    # Keep 256Q students first, then 25B students
    absent_students.sort(
        key=lambda x: (
            0 if str(x["student_id"]).startswith("256Q") else 1,
            str(x["student_id"])
        )
    )

    total_students = len(students)

    present_count = len(present_records)

    absent_count = len(absent_students)

    if total_students > 0:

        attendance_percentage = round(
            (present_count / total_students) * 100,
            2
        )

    else:

        attendance_percentage = 0

    connection.close()

    return render_template(
        "reports.html",
        today=selected_date,
        present_records=present_records,
        absent_students=absent_students,
        total_students=total_students,
        present_count=present_count,
        absent_count=absent_count,
        attendance_percentage=attendance_percentage
    )
    
@app.route("/monthly-reports")
def monthly_reports():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    current_month = request.args.get(
    "month",
    datetime.now().strftime("%Y-%m")
)

    # Count the total class/attendance days in this month
    cursor.execute(
        """
        SELECT COUNT(DISTINCT date) AS total_days
        FROM attendance
        WHERE date LIKE ?
        """,
        (current_month + "%",)
    )

    result = cursor.fetchone()
    total_days = result["total_days"] if result else 0

    report_data = []

    # Include ALL students
    for student_id, student in students.items():

        cursor.execute(
            """
            SELECT COUNT(*) AS present_days
            FROM attendance
            WHERE student_id = ?
              AND date LIKE ?
              AND status = 'Present'
            """,
            (student_id, current_month + "%")
        )

        result = cursor.fetchone()

        present_days = result["present_days"] if result else 0

        absent_days = max(total_days - present_days, 0)

        if total_days > 0:
            attendance_percentage = round(
                (present_days / total_days) * 100,
                2
            )
        else:
            attendance_percentage = 0

        report_data.append({
            "student_id": student_id,
            "student_name": student.get("name", student_id),
            "present_days": present_days,
            "absent_days": absent_days,
            "attendance_percentage": attendance_percentage
        })

    connection.close()

    # Keep 256Q students first, then 25B students
    report_data.sort(
        key=lambda x: (
            0 if str(x["student_id"]).startswith("256Q") else 1,
            str(x["student_id"])
        )
    )

    return render_template(
        "monthly_reports.html",
        month=current_month,
        reports=report_data,
        total_days=total_days
    )

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    current_month = datetime.now().strftime("%Y-%m")

    cursor.execute(
        """
        SELECT student_id,
               student_name,
               COUNT(*) AS present_days
        FROM attendance
        WHERE date LIKE ?
          AND status = 'Present'
        GROUP BY student_id
        ORDER BY student_id
        """,
        (current_month + "%",)
    )

    records = cursor.fetchall()

    report_data = []

    for record in records:

        student_id = str(record["student_id"])

        report_data.append({
            "student_id": student_id,
            "student_name": record["student_name"],
            "present_days": record["present_days"]
        })

    connection.close()

    return render_template(
        "monthly_reports.html",
        month=current_month,
        reports=report_data
    )
@app.route("/export-monthly-report")
def export_monthly_report():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    selected_month = request.args.get(
        "month",
        datetime.now().strftime("%Y-%m")
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(DISTINCT date) AS total_days
        FROM attendance
        WHERE date LIKE ?
        """,
        (selected_month + "%",)
    )

    result = cursor.fetchone()
    total_days = result["total_days"] if result else 0

    report_data = []

    for student_id, student in students.items():

        cursor.execute(
            """
            SELECT COUNT(*) AS present_days
            FROM attendance
            WHERE student_id = ?
              AND date LIKE ?
              AND status = 'Present'
            """,
            (student_id, selected_month + "%")
        )

        result = cursor.fetchone()

        present_days = result["present_days"] if result else 0
        absent_days = max(total_days - present_days, 0)

        if total_days > 0:
            attendance_percentage = round(
                (present_days / total_days) * 100,
                2
            )
        else:
            attendance_percentage = 0

        report_data.append({
            "student_id": student_id,
            "student_name": student.get("name", student_id),
            "present_days": present_days,
            "absent_days": absent_days,
            "attendance_percentage": attendance_percentage
        })

    connection.close()

    report_data.sort(
        key=lambda x: (
            0 if str(x["student_id"]).startswith("256Q") else 1,
            str(x["student_id"])
        )
    )

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Monthly Attendance"

    sheet.append([
        "S.No",
        "Student ID",
        "Student Name",
        "Present Days",
        "Absent Days",
        "Attendance %"
    ])

    for index, student in enumerate(report_data, start=1):

        sheet.append([
            index,
            student["student_id"],
            student["student_name"],
            student["present_days"],
            student["absent_days"],
            student["attendance_percentage"]
        ])

    file_name = f"attendance_report_{selected_month}.xlsx"

    file_path = os.path.join(
        ATTENDANCE_PATH,
        file_name
    )

    workbook.save(file_path)

    return send_file(
        file_path,
        as_attachment=True,
        download_name=file_name
    )

@app.route("/students")
def student_management():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    student_list = []

    for student_id, student in students.items():

        student_list.append({
            "student_id": student_id,
            "name": student.get("name", student_id),
            "department": student.get("department", "AID"),
            "year": student.get("year", "2ND YEAR"),
            "section": student.get("section", "DAYSCHOLAR")
        })

    # Sort students by ID
    student_list.sort(
        key=lambda x: (
            0 if str(x["student_id"]).startswith("256Q") else 1,
            str(x["student_id"])
        )
    )

    return render_template(
        "student_management.html",
        students=student_list,
        total_students=len(student_list)
    )
@app.route("/export-daily-report")
def export_daily_report():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    selected_date = request.args.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT student_id,
               student_name,
               MIN(time) AS time,
               status
        FROM attendance
        WHERE date = ?
        GROUP BY student_id
        """,
        (selected_date,)
    )

    attendance_records = cursor.fetchall()

    attendance_by_student = {}

    for record in attendance_records:

        attendance_by_student[
            str(record["student_id"])
        ] = {
            "student_name": record["student_name"],
            "time": record["time"],
            "status": record["status"]
        }

    report_data = []

    # Include ALL students
    for student_id, student in students.items():

        student_id = str(student_id)

        if student_id in attendance_by_student:

            record = attendance_by_student[student_id]

            status = record["status"]
            time = record["time"]

        else:

            status = "Absent"
            time = "-"

        report_data.append({
            "student_id": student_id,
            "student_name": student.get(
                "name",
                student_id
            ),
            "status": status,
            "time": time
        })

    connection.close()

    # Keep 256Q students first, then 25B students
    report_data.sort(
        key=lambda x: (
            0 if str(x["student_id"]).startswith("256Q") else 1,
            str(x["student_id"])
        )
    )

    # Create Excel workbook
    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "Daily Attendance"

    sheet.append([
        "S.No",
        "Student ID",
        "Student Name",
        "Status",
        "Time"
    ])

    for index, student in enumerate(
        report_data,
        start=1
    ):

        sheet.append([
            index,
            student["student_id"],
            student["student_name"],
            student["status"],
            student["time"]
        ])

    file_name = (
        f"daily_attendance_{selected_date}.xlsx"
    )

    file_path = os.path.join(
        ATTENDANCE_PATH,
        file_name
    )

    workbook.save(file_path)

    return send_file(
        file_path,
        as_attachment=True,
        download_name=file_name
    )
  
# ==========================================
# STUDENT DASHBOARD
# ==========================================

@app.route("/student")
def student_dashboard():

    if session.get("user_type") != "student":
        return redirect(url_for("login"))

    student_id = session.get("student_id")

    if student_id not in students:
        return redirect(url_for("login"))

    student = students[student_id]

    # Get selected month
    selected_month = request.args.get(
        "month",
        datetime.now().strftime("%Y-%m")
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    # Get this student's attendance records
    cursor.execute(
        """
        SELECT date,
               time,
               status
        FROM attendance
        WHERE student_id = ?
          AND date LIKE ?
        ORDER BY date DESC, time DESC
        """,
        (
            student_id,
            selected_month + "%"
        )
    )

    attendance_records = cursor.fetchall()

    # Count this student's present days
    present_days = 0

    for record in attendance_records:

        if record["status"].lower() == "present":
            present_days += 1

    # Find the total number of attendance/working days
    # recorded for the selected month across all students
    cursor.execute(
        """
        SELECT COUNT(DISTINCT date) AS total_days
        FROM attendance
        WHERE date LIKE ?
        """,
        (selected_month + "%",)
    )

    result = cursor.fetchone()

    total_days = result["total_days"] if result else 0

    # Calculate absent days
    absent_days = max(
        total_days - present_days,
        0
    )

    # Calculate attendance percentage
    if total_days > 0:

        attendance_percentage = round(
            (present_days / total_days) * 100,
            2
        )

    else:

        attendance_percentage = 0

    connection.close()

    return render_template(
        "student_dashboard.html",
        student=student,
        student_id=student_id,
        attendance_records=attendance_records,
        present_days=present_days,
        absent_days=absent_days,
        attendance_percentage=attendance_percentage,
        selected_month=selected_month
    )
@app.route("/edit-student-name/<student_id>", methods=["GET", "POST"])
def edit_student_name(student_id):

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    # Check whether student exists
    if student_id not in students:
        return redirect(url_for("student_management"))

    if request.method == "POST":

        new_name = request.form["name"].strip()

        if new_name:

            # Update name in memory
            students[student_id]["name"] = new_name

            # Save updated student database
            with open(STUDENTS_PATH, "w") as file:
                json.dump(students, file, indent=4)

            return redirect(url_for("student_management"))

    return render_template(
        "edit_student_name.html",
        student_id=student_id,
        student=students[student_id]
    )
@app.route("/delete-student/<student_id>", methods=["POST"])
def delete_student(student_id):

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    if student_id not in students:
        return redirect(url_for("student_management"))

    # Student IDs are validated on creation. Keep legacy malformed data from
    # ever being interpreted as a filesystem path.
    if not valid_student_id(student_id):
        return redirect(url_for("student_management"))

    # Remove student from students.json
    del students[student_id]

    with open(STUDENTS_PATH, "w") as file:
        json.dump(students, file, indent=4)

    student_folder = os.path.join(DATASET_PATH, student_id)

    if os.path.exists(student_folder):
        shutil.rmtree(student_folder)

    # Retrain before returning so the removed person cannot still be recognised
    # by the in-memory model.
    try:
        subprocess.run(
            [os.path.join(BASE_DIR, "venv", "bin", "python"), "train_model.py"],
            cwd=BASE_DIR,
            check=True,
        )
        if not reload_recognition_assets():
            print("⚠️ Face model trained but could not be reloaded.")
    except subprocess.CalledProcessError as error:
        print("❌ Face model retraining failed:", error)

    return redirect(url_for("student_management"))

@app.route("/mobile-register")
def mobile_register():
    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    student_id = session.get("registration_student_id")
    student_name = session.get("registration_student_name")

    if not student_id or not student_name:
        return redirect(url_for("student_management"))

    return render_template(
        "mobile_register.html",
        student_id=student_id,
        student_name=student_name,
        capture_target=FACE_IMAGES_REQUIRED
    )
@app.route("/api/register-face", methods=["POST"])
def register_face():

    global recognizer
    global labels

    if session.get("user_type") != "admin":
        return jsonify({
            "success": False,
            "message": "Unauthorized"
        }), 403

    student_id = session.get("registration_student_id")

    if not student_id:
        return jsonify({
            "success": False,
            "message": "Registration session expired."
        }), 400

    data = request.get_json(silent=True)

    if not data or "image" not in data:
        return jsonify({
            "success": False,
            "message": "No image received."
        }), 400

    try:
        image_bytes = decode_camera_image(data["image"])

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_GRAYSCALE
        )

        if frame is None:
            return jsonify({
                "success": False,
                "message": "Invalid image."
            }), 400

        faces = face_detector.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(120, 120)
        )

        if len(faces) != 1:
            return jsonify({
                "success": False,
                "message": "Please keep exactly one face in the camera."
            }), 400

        x, y, w, h = faces[0]

        face = frame[y:y + h, x:x + w]

        face = cv2.resize(
            face,
            (200, 200)
        )

        dataset_folder = os.path.join(DATASET_PATH, student_id)

        os.makedirs(
            dataset_folder,
            exist_ok=True
        )

        existing_images = [
            file
            for file in os.listdir(dataset_folder)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        if len(existing_images) >= FACE_IMAGES_REQUIRED:
            return jsonify({
                "success": True,
                "message": "Face registration is already complete.",
                "count": len(existing_images),
                "training_completed": True
            })

        image_number = len(existing_images) + 1

        image_path = os.path.join(
            dataset_folder,
            f"{image_number:03d}.jpg"
        )

        saved = cv2.imwrite(
            image_path,
            face
        )

        if not saved:
            return jsonify({
                "success": False,
                "message": "Failed to save face image."
            }), 500

        print(
            f"✅ Face image saved: "
            f"{student_id} - {image_number}/{FACE_IMAGES_REQUIRED}"
        )

        # Automatically train once the registration target is reached.

        if image_number >= FACE_IMAGES_REQUIRED:

            print(
                f"🎓 {student_id} reached {FACE_IMAGES_REQUIRED} images."
            )

            try:

                python_path = os.path.join(BASE_DIR, "venv", "bin", "python")
                train_script = os.path.join(BASE_DIR, "train_model.py")

                subprocess.run(
                    [
                        python_path,
                        train_script
                    ], cwd=BASE_DIR,
                    check=True
                )

                print(
                    "✅ Face model automatically trained."
                )

                if not reload_recognition_assets():
                    raise RuntimeError("The trained model could not be reloaded.")

                print(
                    "✅ New face model loaded."
                )

                return jsonify({
                    "success": True,
                    "message": (
                        f"🎉 {FACE_IMAGES_REQUIRED} images captured. "
                        "Face model trained successfully!"
                    ),
                    "count": image_number,
                    "training_completed": True
                })

            except Exception as error:

                print(
                    "❌ Automatic training failed:",
                    error
                )

                return jsonify({
                    "success": True,
                    "message": (
                        f"{FACE_IMAGES_REQUIRED} images captured, "
                        "but model training failed."
                    ),
                    "count": image_number,
                    "training_completed": False
                })

        # ==========================================
        # IMAGE SAVED
        # ==========================================

        return jsonify({
            "success": True,
            "message": "Face image saved.",
            "count": image_number,
            "training_completed": False
        })

    except ValueError as error:

        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except Exception as error:

        print(
            "❌ Face registration error:",
            error
        )

        return jsonify({
            "success": False,
            "message": "Failed to save face image."
        }), 500

@app.route("/add-student", methods=["GET", "POST"])
def add_student():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    if request.method == "POST":

        student_id = request.form["student_id"].strip()
        name = request.form["name"].strip()
        department = request.form["department"].strip()
        year = request.form["year"].strip()
        section = request.form["section"].strip()

        if not student_id or not name:
            return render_template("add_student.html")

        if not valid_student_id(student_id):
            return render_template(
                "add_student.html",
                error="Student ID may contain only letters, numbers, spaces, hyphens, and underscores."
            )

        # Check if student already exists
        if student_id in students:
            return render_template(
                "add_student.html",
                error="Student ID already exists."
            )

        # Save student details
        students[student_id] = {
            "name": name,
            "department": department,
            "year": year,
            "section": section
        }

        with open(STUDENTS_PATH, "w") as file:
            json.dump(students, file, indent=4)

        # Open mobile face registration page
        session["registration_student_id"] = student_id
        session["registration_student_name"] = name

        return redirect(url_for("mobile_register"))

    return render_template("add_student.html")

# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ==========================================
# START FLASK
# ==========================================

if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() == "true",
        ssl_context="adhoc"
    )

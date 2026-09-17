from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import os
import json
import sqlite3
import subprocess
import base64
import numpy as np
import cv2
from datetime import datetime
from openpyxl import Workbook

app = Flask(__name__)

# ==========================================
# SECRET KEY
# ==========================================

app.secret_key = "ai-attendance-secret-key"


# ==========================================
# FILE PATHS
# ==========================================

STUDENTS_PATH = "students.json"
DATABASE_PATH = "attendance/attendance.db"
MODEL_PATH = "models/face_model.yml"
LABELS_PATH = "models/labels.json"
CASCADE_PATH = "haarcascade_frontalface_default.xml"


# ==========================================
# LOGIN DETAILS
# ==========================================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


# ==========================================
# LOAD STUDENT DATABASE
# ==========================================

if not os.path.exists(STUDENTS_PATH):
    print("❌ students.json not found.")
    exit()

with open(STUDENTS_PATH, "r") as file:
    students = json.load(file)

print(f"✅ Student database loaded: {len(students)} students")


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

    os.makedirs("attendance", exist_ok=True)

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

@app.route("/start-attendance")
def start_attendance():

    if session.get("user_type") != "admin":
        return redirect(url_for("login"))

    try:

        project_path = os.path.abspath(".")

        python_path = os.path.join(
            project_path,
            "venv",
            "bin",
            "python"
        )

        script_path = os.path.join(
            project_path,
            "attendance_system.py"
        )

        # Build command WITHOUT quotation marks
        command = (
            f"cd {project_path} && "
            f"{python_path} {script_path}"
        )

        print("📷 Starting AI Face Attendance...")
        print(f"Command: {command}")

        # Escape characters for AppleScript
        apple_command = command.replace("\\", "\\\\")
        apple_command = apple_command.replace('"', '\\"')

        # Open a new Terminal window
        subprocess.Popen(
            [
                "osascript",
                "-e",
                f'tell application "Terminal" to do script "{apple_command}"'
            ]
        )

        print("✅ Attendance program launched.")

    except Exception as error:

        print("❌ Could not start attendance system.")
        print(error)

    return redirect(url_for("admin_dashboard"))

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
        "attendance",
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
        "attendance",
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

    # Remove student from students.json
    del students[student_id]

    with open(STUDENTS_PATH, "w") as file:
        json.dump(students, file, indent=4)

    # Remove student's face dataset
    student_folder = os.path.join(
        "dataset",
        student_id
    )

    if os.path.exists(student_folder):
        import shutil

        shutil.rmtree(student_folder)

    # Automatically retrain the face-recognition model
    subprocess.Popen([
        "venv/bin/python",
        "train_model.py"
    ])

    return redirect(url_for("student_management"))
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

        # Start face registration
        subprocess.Popen([
            "venv/bin/python",
            "register_student.py",
            student_id,
            name
        ])

        return redirect(url_for("student_management"))

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
        debug=True
    )
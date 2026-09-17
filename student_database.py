import sqlite3
import os

# Create attendance folder
os.makedirs("attendance", exist_ok=True)

# Connect to database
connection = sqlite3.connect("attendance/attendance.db")
cursor = connection.cursor()

# Create students table
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    student_name TEXT NOT NULL,
    department TEXT,
    year TEXT,
    section TEXT
)
""")

connection.commit()

print("\n================================")
print("       STUDENT REGISTRATION")
print("================================")

student_id = input("Enter Student ID: ").strip()
student_name = input("Enter Student Name: ").strip()
department = input("Enter Department: ").strip()
year = input("Enter Year: ").strip()
section = input("Enter Section: ").strip()

if not student_id or not student_name:
    print("\n❌ Student ID and Name are required.")
    connection.close()
    exit()

try:
    cursor.execute("""
    INSERT INTO students
    (student_id, student_name, department, year, section)
    VALUES (?, ?, ?, ?, ?)
    """, (
        student_id,
        student_name,
        department,
        year,
        section
    ))

    connection.commit()

    print("\n✅ STUDENT REGISTERED")
    print("--------------------------------")
    print(f"ID         : {student_id}")
    print(f"Name       : {student_name}")
    print(f"Department : {department}")
    print(f"Year       : {year}")
    print(f"Section    : {section}")
    print("--------------------------------")

except sqlite3.IntegrityError:
    print("\n⚠️ Student ID already exists.")

connection.close()

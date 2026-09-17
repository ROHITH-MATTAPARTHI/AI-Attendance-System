import os
import json
import re

PHOTO_FOLDER = "students photos"
OUTPUT_FILE = "students.json"

students = {}

# Existing students
students["25B21A4533"] = {
    "name": "ROHITH MATTAPARTHI",
    "department": "AID",
    "year": "2ND YEAR",
    "section": "DAYSCHOLAR"
}

students["25B21A4508"] = {
    "name": "VENKATA SRINIDHI GAMINI",
    "department": "AID",
    "year": "2ND YEAR",
    "section": "DAYSCHOLAR"
}

# Add students from photo filenames
for filename in os.listdir(PHOTO_FOLDER):

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    student_id = os.path.splitext(filename)[0]

    # Remove duplicate suffix such as (2)
    student_id = re.sub(r"\s*\(\d+\)$", "", student_id)

    if student_id not in students:
        students[student_id] = {
            "name": student_id,
            "department": "AID",
            "year": "2ND YEAR",
            "section": "DAYSCHOLAR"
        }

with open(OUTPUT_FILE, "w") as file:
    json.dump(students, file, indent=4)

print("================================")
print("STUDENT DATABASE CREATED")
print("================================")
print(f"Total students: {len(students)}")
print(f"Saved to: {OUTPUT_FILE}")
print("================================")
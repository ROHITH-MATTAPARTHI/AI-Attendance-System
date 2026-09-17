import os
import subprocess

def show_menu():
    print("\n")
    print("========================================")
    print("        🤖 AI ATTENDANCE SYSTEM")
    print("========================================")
    print("1. Register Student")
    print("2. Train Face Recognition Model")
    print("3. Start Attendance")
    print("4. View Attendance Database")
    print("5. Exit")
    print("========================================")


while True:

    show_menu()

    choice = input("Enter your choice: ").strip()

    if choice == "1":
        print("\n📚 Starting Student Registration...\n")
        subprocess.run(["python", "student_database.py"])
        subprocess.run(["python", "register_student.py"])

    elif choice == "2":
        print("\n🧠 Training Face Recognition Model...\n")
        subprocess.run(["python", "train_model.py"])

    elif choice == "3":
        print("\n📷 Starting AI Attendance System...\n")
        subprocess.run(["python", "attendance_system.py"])

    elif choice == "4":
        print("\n📊 ATTENDANCE RECORDS")
        print("========================================")

        if os.path.exists("attendance/attendance.db"):
            subprocess.run([
                "sqlite3",
                "attendance/attendance.db",
                "SELECT * FROM attendance;"
            ])
        else:
            print("❌ Attendance database not found.")

    elif choice == "5":
        print("\n👋 Exiting AI Attendance System.")
        break

    else:
        print("\n❌ Invalid choice. Please enter 1-5.")
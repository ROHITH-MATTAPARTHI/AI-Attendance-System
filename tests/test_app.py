import unittest

import app


class AttendanceAppTests(unittest.TestCase):
    def setUp(self):
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    def as_admin(self):
        with self.client.session_transaction() as session:
            session["user_type"] = "admin"

    def test_protected_pages_redirect_without_login(self):
        for path in ("/admin", "/reports", "/students", "/student"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302)
            self.assertTrue(response.location.endswith("/"))

    def test_admin_pages_render(self):
        self.as_admin()
        for path in ("/admin", "/reports", "/monthly-reports", "/students"):
            self.assertEqual(self.client.get(path).status_code, 200)

        dashboard = self.client.get("/admin")
        self.assertIn(b'class="attendance-table"', dashboard.data)
        self.assertIn(b"table-layout: fixed", dashboard.data)

    def test_invalid_student_id_is_rejected_without_writing(self):
        self.as_admin()
        before = set(app.students)
        response = self.client.post(
            "/add-student",
            data={
                "student_id": "../../outside",
                "name": "Invalid ID",
                "department": "AID",
                "year": "2ND YEAR",
                "section": "DAYSCHOLAR",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student ID may contain", response.data)
        self.assertEqual(set(app.students), before)

    def test_invalid_camera_payload_returns_client_error(self):
        self.as_admin()
        response = self.client.post("/api/recognize", json={"image": ["not-a-string"]})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json["attendance_marked"])

    def test_recognition_assets_loaded(self):
        self.assertIsNotNone(app.recognizer)
        self.assertIsNotNone(app.face_detector)
        self.assertTrue(app.labels)

    def test_registration_page_uses_configured_capture_target(self):
        self.as_admin()
        with self.client.session_transaction() as session:
            session["registration_student_id"] = "roxx"
            session["registration_student_name"] = "roxx45"
        response = self.client.get("/mobile-register")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"0 / 20", response.data)
        self.assertIn(b"const TOTAL_IMAGES = 20", response.data)
        self.assertIn(b'window.location.href = "/students"', response.data)


if __name__ == "__main__":
    unittest.main()

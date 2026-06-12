import os
import sys
import unittest
import requests

BASE_URL = "http://127.0.0.1:8000"

class TestCVAnalyzerAPI(unittest.TestCase):
    def setUp(self):
        # Verify if the server is up
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            self.server_online = r.status_code == 200
        except requests.exceptions.ConnectionError:
            self.server_online = False
        
        if not self.server_online:
            self.skipTest("FastAPI server is offline at http://127.0.0.1:8000. Start it first.")

    def test_01_health_endpoint(self):
        print("\nTesting GET /health...")
        response = requests.get(f"{BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        print(f"Health check status: {data.get('status')}, version: {data.get('version')}")

    def test_02_upload_and_analyze(self):
        print("\nTesting POST /api/analyze/upload...")
        # Resolve sample CV file path
        sample_dir = os.path.join(os.path.dirname(__file__), "..", "Sample_CVs")
        sample_cv_path = os.path.join(sample_dir, "Finance Sample Resume.pdf")
        
        if not os.path.exists(sample_cv_path):
            # Try finding any pdf file in Sample_CVs
            if os.path.exists(sample_dir):
                pdf_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".pdf")]
                if pdf_files:
                    sample_cv_path = os.path.join(sample_dir, pdf_files[0])
            
        if not os.path.exists(sample_cv_path):
            self.skipTest(f"No sample CV PDF found at {sample_cv_path} to test upload.")

        print(f"Uploading file: {os.path.basename(sample_cv_path)}")
        
        with open(sample_cv_path, "rb") as f:
            files = {"file": (os.path.basename(sample_cv_path), f, "application/pdf")}
            data = {
                "target_jd": "Financial analyst with experience in financial modeling, budgeting, and forecasting.",
                "user_id": 999  # Temporary test user ID
            }
            response = requests.post(f"{BASE_URL}/api/analyze/upload", files=files, data=data)
            
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        self.assertIn("id", result)
        self.assertIn("predicted_role", result)
        self.assertIn("confidence", result)
        self.assertIn("ats_score", result)
        
        print(f"Analysis successful! ID: {result['id']}")
        print(f"Predicted Role: {result['predicted_role']} (Confidence: {result['confidence']:.2f})")
        print(f"ATS Score: {result['ats_score']}%")
        print(f"Extracted Skills: {result.get('extracted_skills')}")
        print(f"Missing Skills: {result.get('missing_skills')}")
        
        # Save ID for later tests
        self.__class__.created_analysis_id = result["id"]
        self.__class__.sample_cv_text = result.get("cv_text")

    def test_03_history_list(self):
        print("\nTesting GET /api/history/...")
        response = requests.get(f"{BASE_URL}/api/history/", params={"user_id": 999})
        self.assertEqual(response.status_code, 200)
        history = response.json()
        self.assertIsInstance(history, list)
        self.assertTrue(len(history) > 0, "History should not be empty after successful upload")
        print(f"Found {len(history)} records in history for user 999.")
        
        # Verify the record matches our created one
        created_id = getattr(self.__class__, "created_analysis_id", None)
        if created_id:
            found_ids = [item["id"] for item in history]
            self.assertIn(created_id, found_ids)

    def test_04_history_detail(self):
        created_id = getattr(self.__class__, "created_analysis_id", None)
        if not created_id:
            self.skipTest("No analysis record was created in prior test step.")
            
        print(f"\nTesting GET /api/history/{created_id}...")
        response = requests.get(f"{BASE_URL}/api/history/{created_id}")
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertEqual(detail["id"], created_id)
        self.assertIn("cv_filename", detail)
        self.assertIn("all_scores", detail)
        self.assertIn("tips", detail)
        print(f"Retrieved detail for analysis ID {created_id} successfully.")

    def test_05_rewrite(self):
        cv_text = getattr(self.__class__, "sample_cv_text", None)
        if not cv_text:
            cv_text = "Jane Doe\nFinance Professional\nBudgeting, financial forecasting, Excel modeling."
            
        print("\nTesting POST /api/rewrite/...")
        payload = {
            "cv_text": cv_text,
            "job_description": "We are seeking a senior financial analyst skilled in financial modeling, forecasting, budgeting, and PowerPoint presentation."
        }
        response = requests.post(f"{BASE_URL}/api/rewrite/", json=payload)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("rewritten_cv", result)
        self.assertIn("old_ats_score", result)
        self.assertIn("new_ats_score", result)
        print("CV Rewriting test completed.")
        print(f"Old ATS Score: {result['old_ats_score']}%")
        print(f"New ATS Score: {result['new_ats_score']}%")
        
        # Save rewritten CV for download test
        self.__class__.rewritten_cv_text = result["rewritten_cv"]

    def test_06_rewrite_download(self):
        rewritten_text = getattr(self.__class__, "rewritten_cv_text", None)
        if not rewritten_text:
            self.skipTest("No rewritten CV text available from test_05.")
            
        print("\nTesting POST /api/rewrite/download...")
        payload = {"rewritten_cv": rewritten_text}
        response = requests.post(f"{BASE_URL}/api/rewrite/download", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        self.assertTrue(len(response.content) > 0)
        print(f"Downloaded DOCX document successfully (size: {len(response.content)} bytes).")

    def test_07_delete_history(self):
        created_id = getattr(self.__class__, "created_analysis_id", None)
        if not created_id:
            self.skipTest("No analysis record was created in prior test step.")
            
        print(f"\nTesting DELETE /api/history/{created_id}...")
        response = requests.delete(f"{BASE_URL}/api/history/{created_id}")
        self.assertEqual(response.status_code, 200)
        
        # Double check it is deleted
        check_response = requests.get(f"{BASE_URL}/api/history/{created_id}")
        self.assertEqual(check_response.status_code, 404)
        print(f"Successfully cleaned up analysis record ID {created_id}.")

if __name__ == "__main__":
    unittest.main()

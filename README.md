# Senior_citizen_task-Task-2
An AI video tracking system built with Streamlit and OpenCV that counts unique customers, filters out background shadows, and logs their age, gender, and senior citizen status to Excel.

# 🛒 Smart Retail Person Tracking & Excel Logging System

A simple AI tool that tracks unique customers in store videos, finds their age/gender, and saves the data cleanly to an Excel sheet.

---

## ❓ Problem Statement
CCTV videos in busy stores are often blurry, have dark shadows, and show people from difficult side-profile angles. Standard AI systems make two major mistakes under these conditions:
1. They misidentify shelf shadows or empty spaces as real people.
2. They log the same person multiple times (creating duplicate rows) or guess a completely wrong age when a person looks away from the camera.

This project solves these issues by tracking people smoothly across frames and cleaning up shadow noise so that **one real person creates exactly one correct row** in the database.

---

## 📊 Dataset & Models
This system runs completely offline on your local machine using six project files:
* **Face Detection Engine:** `deploy.prototxt` & `res10_300x300_ssd_iter_140000.caffemodel` (Finds where a face is located).
* **Age & Gender Engine:** `age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt`, & `gender_net.caffemodel` (Pre-trained on the standard Adience dataset to classify groups).

---

## 🛠️ Methodology (How It Works)
The program processes your video stream in three simple steps:
1. **High Sensitivity Scanning:** The system looks for faces at a low threshold. Even if only a tiny portion or a sharp side-view of a face is visible, it locks on and creates a unique Person ID.
2. **Smart Shadow Filtering:** As the person moves, the code ignores dark frames or bad angles. It checks how many frames a profile is seen; if it's a random background flash or shadow, it deletes it automatically.
3. **Most-Frequent Voting Engine:** When the video ends, the system checks all the clear moments captured for that specific ID. It picks the most frequent age and gender group recorded, ensuring a highly stable final value.

---

## 📊 Expected Results
The final results are appended cleanly into a local file named `customer_visit_log.xlsx`. 

### Example Excel Output Table:
| Time of Visit | Source Video | Predicted Age | Gender | Senior Citizen Status |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-20 14:22:05 | entrance_cam.mp4 | 28 | Male | Not Senior Citizen |
| 2026-06-20 14:23:41 | aisle_04_video.mov | 72 | Female | Senior Citizen |

---

## 🚀 How to Run the App
1. Install requirements: `pip install -r requirements.txt`
2. Start the app: `streamlit run senior_citizen_task.py`
3. Upload your store videos into the browser dashboard and look at your generated Excel file!

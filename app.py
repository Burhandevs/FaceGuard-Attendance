from flask import Flask, render_template, request, session
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, date
import sqlite3
import json
import pandas as pd
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils
from sklearn.metrics import precision_score, f1_score

import time

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Initialize performance metrics database
def init_performance_db():
    conn = sqlite3.connect('recognition_metrics.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS RecognitionMetrics 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     name TEXT,
                     precision REAL,
                     f1_score REAL,
                     timestamp TEXT)''')
    conn.commit()
    conn.close()

init_performance_db()

# Eye aspect ratio calculation for liveness detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Calculate mouth aspect ratio for liveness detection
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Calculate head pose (rudimentary using face landmarks)
def head_pose(shape):
    nose = shape[30]
    left_eye = np.mean([shape[36], shape[37], shape[38], shape[39]], axis=0)
    right_eye = np.mean([shape[42], shape[43], shape[44], shape[45]], axis=0)
    d1 = dist.euclidean(nose, left_eye)
    d2 = dist.euclidean(nose, right_eye)
    if d1 > d2 + 5:
        return "left"
    elif d2 > d1 + 5:
        return "right"
    return "center"

@app.route('/new', methods=['GET', 'POST'])
def new():
    if request.method == "POST":
        return render_template('StudentRegister.html')
    else:
        return "Everything is okay!"

@app.route('/name', methods=['GET', 'POST'])
def name():
    if request.method == "POST":
        name1 = request.form['name1']
        name2 = request.form['name2']
        username_folder = f'Training images/{name1}'

        # First check if name already exists in database
        conn = sqlite3.connect('information.db')
        cursor = conn.cursor()
        cursor.execute("SELECT NAME FROM Users WHERE NAME=?", (name1,))
        existing_name = cursor.fetchone()
        conn.close()
        
        if existing_name:
            return "Name already registered in the system!"

        # Now check face similarity with existing images
        existing_encodings = []
        existing_names = []
        
        for existing_folder in os.listdir('Training images'):
            for existing_file in os.listdir(os.path.join('Training images', existing_folder)):
                existing_img_path = os.path.join('Training images', existing_folder, existing_file)
                existing_img = cv2.imread(existing_img_path)
                if existing_img is not None:
                    existing_encode = face_recognition.face_encodings(existing_img)
                    if existing_encode:
                        existing_encodings.append(existing_encode[0])
                        existing_names.append(existing_folder)

        # Create folder for new user (temporarily)
        os.makedirs(username_folder, exist_ok=True)

        # Capture new images for comparison
        cam = cv2.VideoCapture(0)
        img_count = 0
        total_images = 10
        captured_images = []
        captured_encodings = []

        while img_count < total_images:
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break

            text = f"Captured: {img_count}/{total_images} images"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Press Space to capture image", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                img_name = f"{name1}_{img_count}.png"
                img_path = os.path.join(username_folder, img_name)
                cv2.imwrite(img_path, frame)
                captured_images.append(frame)
                
                # Get encoding of captured image
                current_encode = face_recognition.face_encodings(frame)
                if current_encode:
                    captured_encodings.append(current_encode[0])
                
                print(f"{img_name} written!")
                img_count += 1

        cam.release()
        cv2.destroyAllWindows()

        # Check if face is already registered (regardless of name)
        registered = False
        registered_name = None
        if existing_encodings and captured_encodings:
            for cap_encode in captured_encodings:
                matches = face_recognition.compare_faces(existing_encodings, cap_encode, tolerance=0.5)
                if True in matches:
                    match_idx = matches.index(True)
                    registered_name = existing_names[match_idx]
                    registered = True
                    break

        if registered:
            # Delete the newly captured images since this is a duplicate
            try:
                for img_file in os.listdir(username_folder):
                    os.remove(os.path.join(username_folder, img_file))
                os.rmdir(username_folder)
            except FileNotFoundError:
                pass  # Folder was already deleted or never created
            return f"Face already registered in the system (under name: {registered_name})!"

        # If not registered, proceed with saving to database
        conn = sqlite3.connect('information.db')
        conn.execute('''CREATE TABLE IF NOT EXISTS Users (NAME TEXT)''')
        conn.execute("INSERT OR IGNORE INTO Users (NAME) VALUES (?)", (name1,))
        conn.commit()
        conn.close()

        return render_template('image.html')
    else:
        return 'All is not well'

@app.route("/", methods=["GET", "POST"])
def recognize():
    if request.method == "POST":
        # Initialize databases first
        def initialize_databases():
            # Initialize Attendance database
            conn_attendance = sqlite3.connect('information.db')
            conn_attendance.execute('''CREATE TABLE IF NOT EXISTS Attendance 
                                    (NAME TEXT, Time TEXT, Date TEXT, UNIQUE(NAME, Date))''')
            conn_attendance.commit()
            conn_attendance.close()
            
            # Initialize Metrics database
            conn_metrics = sqlite3.connect('recognition_metrics.db')
            conn_metrics.execute('''CREATE TABLE IF NOT EXISTS RecognitionMetrics 
                                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                   name TEXT,
                                   precision REAL,
                                   recall REAL,
                                   f1_score REAL,
                                   accuracy REAL,
                                   timestamp TEXT)''')
            conn_metrics.commit()
            conn_metrics.close()
        
        initialize_databases()

        # Load training images
        path = 'Training images'
        images = []
        classNames = []
        for folder_name in os.listdir(path):
            folder_path = os.path.join(path, folder_name)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        classNames.append(folder_name)

        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodes = face_recognition.face_encodings(img)
                if encodes:
                    encodeList.append(encodes[0])
            return encodeList

        def check_attendance_status(name):
            today = date.today()
            conn = sqlite3.connect('information.db')
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM Attendance WHERE NAME = ? AND Date = ?", (name, today))
                result = cursor.fetchone()
                return result is not None
            finally:
                conn.close()

        def markData(name):
            if check_attendance_status(name):
                return False
                
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            today = date.today()
            
            conn = sqlite3.connect('information.db')
            try:
                conn.execute("INSERT INTO Attendance (NAME, Time, Date) VALUES (?, ?, ?)", 
                           (name, dtString, today))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False
            finally:
                conn.close()

        def store_recognition_metrics(name, precision, recall, f1, accuracy):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            conn = sqlite3.connect('recognition_metrics.db')
            try:
                conn.execute("INSERT INTO RecognitionMetrics (name, precision, recall, f1_score, accuracy, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                            (name, precision, recall, f1, accuracy, timestamp))
                conn.commit()
            finally:
                conn.close()

        def detect_screen(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(gray, 50, 200)
            
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            screen_detected = False
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    screen_detected = True
                    break
            
            return screen_detected

        encodeListKnown = findEncodings(images)
        known_names = classNames

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        cap = cv2.VideoCapture(0)

        blink_threshold = 0.25
        mar_threshold = 0.5
        head_turn_threshold = 10

        live_frames = 0
        spoof_frames = 0
        consecutive_live_threshold = 5
        consecutive_spoof_threshold = 5
        
        true_labels = []
        predicted_labels = []
        confidence_scores = []
        
        already_marked_timeout = 0
        timeout_started = False

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            if timeout_started:
                current_time = time.time()
                if current_time - already_marked_timeout >= 15:
                    cap.release()
                    cv2.destroyAllWindows()
                    return render_template('first.html')
                continue

            screen_detected = detect_screen(img)
            if screen_detected:
                cv2.putText(img, "SCREEN DETECTED - SPOOFING ATTEMPT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Punch your Attendance', img)
                cv2.waitKey(2000)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            live = False
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                mouth = shape[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)

                nose = shape[30]
                left_eye_center = np.mean([shape[36], shape[37], shape[38], shape[39]], axis=0)
                right_eye_center = np.mean([shape[42], shape[43], shape[44], shape[45]], axis=0)
                d1 = dist.euclidean(nose, left_eye_center)
                d2 = dist.euclidean(nose, right_eye_center)

                head_pose_direction = "center"
                if d1 > d2 + head_turn_threshold:
                    head_pose_direction = "left"
                elif d2 > d1 + head_turn_threshold:
                    head_pose_direction = "right"

                if (ear > blink_threshold and 
                    mar < mar_threshold and 
                    head_pose_direction == "center" and
                    rect.width() > 100):
                    live = True
                else:
                    live = False

                if live:
                    live_frames += 1
                    spoof_frames = 0
                    if live_frames >= consecutive_live_threshold:
                        cv2.putText(img, "Live Person Detected", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                        facesCurFrame = face_recognition.face_locations(imgS)
                        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                            matchIndex = np.argmin(faceDis)
                            
                            true_label = known_names[matchIndex] if matches[matchIndex] else "Unknown"
                            predicted_label = known_names[matchIndex] if faceDis[matchIndex] < 0.50 else "Unknown"
                            
                            true_labels.append(true_label)
                            predicted_labels.append(predicted_label)
                            confidence_scores.append(1 - faceDis[matchIndex])

                            if faceDis[matchIndex] < 0.50:
                                name = known_names[matchIndex].upper()
                                
                                if check_attendance_status(name):
                                    cv2.putText(img, f"{name}: Attendance already marked today", 
                                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    cv2.putText(img, "Closing in 15 seconds...", (10, 120),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    cv2.imshow('Punch your Attendance', img)
                                    if not timeout_started:
                                        already_marked_timeout = time.time()
                                        timeout_started = True
                                    continue
                                
                                if len(set(true_labels)) > 1:
                                    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
                                    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
                                    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
                                    accuracy = accuracy_score(true_labels, predicted_labels)
                                    store_recognition_metrics(name, precision, recall, f1, accuracy)
                                
                                if markData(name):
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    return render_template('first.html')
                            else:
                                name = 'Unknown'

                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, name, (x1 + 6, y2 - 6), 
                                       cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    else:
                        cv2.putText(img, "Checking Liveness...", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                else:
                    spoof_frames += 1
                    live_frames = 0
                    if spoof_frames >= consecutive_spoof_threshold:
                        cv2.putText(img, "Spoofing Suspected", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(img, "Checking Liveness...", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('Punch your Attendance', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return render_template('first.html')
    else:
        return render_template('main.html')

@app.route('/login',methods = ['POST'])
def login():
    json_data = json.loads(request.data.decode())
    username = json_data['username']
    password = json_data['password']
    df= pd.read_csv('cred.csv')
    if len(df.loc[df['username'] == username]['password'].values) > 0:
        if df.loc[df['username'] == username]['password'].values[0] == password:
            session['username'] = username
            return 'success'
        else:
            return 'failed'
    else:
        return 'failed'

@app.route('/checklogin')
def checklogin():
    if 'username' in session:
        return session['username']
    return 'False'

@app.route('/login',methods=["GET","POST"])
def how():
    return render_template('AdminLogin.html')

@app.route('/data',methods=["GET","POST"])
def data():
    if request.method=="POST":
        today=date.today()
        conn = sqlite3.connect('information.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance where Date=?",(today,))
        rows=cur.fetchall()
        conn.close()
        return render_template('TodayAttendance.html',rows=rows)
    else:
        return render_template('AdminLogin.html')

@app.route('/whole',methods=["GET","POST"])
def whole():
    today=date.today()
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance")
    rows=cur.fetchall()
    return render_template('AttendanceList.html',rows=rows)

# @app.route('/metrics')
# def view_metrics():
#     conn = sqlite3.connect('recognition_metrics.db')
#     conn.row_factory = sqlite3.Row
#     cur = conn.cursor()
#     cursor = cur.execute("SELECT name, precision, f1_score, timestamp FROM RecognitionMetrics ORDER BY timestamp DESC")
#     metrics = cur.fetchall()
#     conn.close()
#     return render_template('RecognitionMetrics.html', metrics=metrics)


def initialize_databases():
    # Initialize Metrics database with all required columns
    conn = sqlite3.connect('recognition_metrics.db')
    try:
        # Create table with all columns if it doesn't exist
        conn.execute('''CREATE TABLE IF NOT EXISTS RecognitionMetrics 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT,
                       precision REAL,
                       recall REAL,
                       f1_score REAL,
                       accuracy REAL,
                       timestamp TEXT)''')
        
        # Check for missing columns and add them if needed
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(RecognitionMetrics)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        required_columns = ['precision', 'recall', 'f1_score', 'accuracy']
        for column in required_columns:
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE RecognitionMetrics ADD COLUMN {column} REAL DEFAULT 0")
        
        conn.commit()
    finally:
        conn.close()

@app.route('/metrics')
def view_metrics():
    # First ensure database is properly initialized
    initialize_databases()
    
    conn = sqlite3.connect('recognition_metrics.db')
    conn.row_factory = sqlite3.Row
    
    try:
        # Check which columns exist
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(RecognitionMetrics)")
        columns = [column[1] for column in cur.fetchall()]
        has_columns = {
            'precision': 'precision' in columns,
            'recall': 'recall' in columns,
            'f1_score': 'f1_score' in columns,
            'accuracy': 'accuracy' in columns
        }
        
        # Build dynamic query based on available columns
        select_parts = ["strftime('%Y-%m-%d %H:%M', timestamp) as formatted_time", "name"]
        
        if has_columns['precision']:
            select_parts.append("ROUND(precision*100, 2) as precision_pct")
        else:
            select_parts.append("0 as precision_pct")
            
        if has_columns['recall']:
            select_parts.append("ROUND(recall*100, 2) as recall_pct")
        else:
            select_parts.append("0 as recall_pct")
            
        if has_columns['f1_score']:
            select_parts.append("ROUND(f1_score*100, 2) as f1_pct")
        else:
            select_parts.append("0 as f1_pct")
            
        if has_columns['accuracy']:
            select_parts.append("ROUND(accuracy*100, 2) as accuracy_pct")
        else:
            select_parts.append("0 as accuracy_pct")
        
        query = f"""
            SELECT {', '.join(select_parts)}
            FROM RecognitionMetrics 
            ORDER BY timestamp DESC
            LIMIT 100
        """
        
        cur.execute(query)
        recent_metrics = cur.fetchall()
        
        # Get summary statistics
        avg_parts = []
        if has_columns['precision']:
            avg_parts.append("ROUND(AVG(precision)*100, 2) as avg_precision")
        else:
            avg_parts.append("0 as avg_precision")
            
        if has_columns['recall']:
            avg_parts.append("ROUND(AVG(recall)*100, 2) as avg_recall")
        else:
            avg_parts.append("0 as avg_recall")
            
        if has_columns['f1_score']:
            avg_parts.append("ROUND(AVG(f1_score)*100, 2) as avg_f1")
        else:
            avg_parts.append("0 as avg_f1")
            
        if has_columns['accuracy']:
            avg_parts.append("ROUND(AVG(accuracy)*100, 2) as avg_accuracy")
        else:
            avg_parts.append("0 as avg_accuracy")
        
        summary_query = f"""
            SELECT 
                COUNT(*) as total_recognitions,
                {', '.join(avg_parts)}
            FROM RecognitionMetrics
        """
        
        cur.execute(summary_query)
        summary = cur.fetchone()
        
        # Get metrics by person
        person_query = f"""
            SELECT 
                name,
                COUNT(*) as recognition_count,
                {', '.join(avg_parts)}
            FROM RecognitionMetrics
            GROUP BY name
            ORDER BY recognition_count DESC
        """
        
        cur.execute(person_query)
        metrics_by_person = cur.fetchall()
        
    finally:
        conn.close()
    
    return render_template('RecognitionMetrics.html',
                         recent_metrics=recent_metrics,
                         summary=summary,
                         metrics_by_person=metrics_by_person)

if __name__ == '__main__':
    app.run(debug=True)
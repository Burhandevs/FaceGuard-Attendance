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
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

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
        return render_template('index.html')
    else:
        return "Everything is okay!"

@app.route('/name', methods=['GET', 'POST'])
def name():
    if request.method == "POST":
        name1 = request.form['name1']
        name2 = request.form['name2']
        username_folder = f'Training images/{name1}'

        if not os.path.exists(username_folder):
            os.makedirs(username_folder)

        cam = cv2.VideoCapture(0)
        img_count = 0
        total_images = 10

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
                cv2.imwrite(os.path.join(username_folder, img_name), frame)
                print(f"{img_name} written!")
                img_count += 1

        cam.release()
        cv2.destroyAllWindows()

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

        def markData(name):
            now = datetime.now()
            dtString = now.strftime('%H:%M')
            today = date.today()
            conn = sqlite3.connect('information.db')
            conn.execute('''CREATE TABLE IF NOT EXISTS Attendance (NAME TEXT, Time TEXT, Date TEXT)''')
            conn.execute("INSERT OR IGNORE INTO Attendance (NAME, Time, Date) VALUES (?, ?, ?)", (name, dtString, today))
            conn.commit()
            conn.close()

        def markAttendance(name):
            with open('attendance.csv', 'a+', errors='ignore') as f:
                f.seek(0)
                myDataList = f.readlines()
                nameList = [line.split(',')[0] for line in myDataList]
                if name not in nameList:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M')
                    f.write(f'\n{name},{dtString}')

        encodeListKnown = findEncodings(images)

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

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break

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

                if ear > blink_threshold and mar < mar_threshold and head_pose_direction == "center":
                    live = True
                else:
                    live = False

            if live:
                live_frames += 1
                spoof_frames = 0
                if live_frames >= consecutive_live_threshold:
                    cv2.putText(img, "Live Person Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                    facesCurFrame = face_recognition.face_locations(imgS)
                    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        matchIndex = np.argmin(faceDis)

                        if faceDis[matchIndex] < 0.50:
                            name = classNames[matchIndex].upper()
                            markAttendance(name)
                            markData(name)
                            cap.release()
                            cv2.destroyAllWindows()
                            return render_template('first.html')
                        else:
                            name = 'Unknown'

                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(img, "Checking Liveness...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            else:
                spoof_frames += 1
                live_frames = 0
                if spoof_frames >= consecutive_spoof_threshold:
                    cv2.putText(img, "Spoofing Suspected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(img, "Checking Liveness...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('Punch your Attendance', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return render_template('first.html')
    else:
        return render_template('main.html')

# Remaining routes and login handling are unchanged.
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

@app.route('/how',methods=["GET","POST"])
def how():
    return render_template('form.html')

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
        return render_template('form2.html',rows=rows)
    else:
        return render_template('form1.html')

@app.route('/whole',methods=["GET","POST"])
def whole():
    today=date.today()
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance")
    rows=cur.fetchall()
    return render_template('form3.html',rows=rows)

if __name__ == '__main__':
    app.run(debug=True)

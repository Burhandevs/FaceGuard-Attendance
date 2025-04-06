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

# Eye aspect ratio calculation for spoofing detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

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

        cam = cv2.VideoCapture(0)
        img_count = 0  # Count how many images we've captured
        total_images = 10  # Total images to be captured

        while img_count < total_images:  # Capture 10 images
            ret, frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break

            # Display the current image with the counter overlay
            text = f"Captured: {img_count}/{total_images} images"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Press Space to capture image", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC key to exit
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:  # Space key to capture image
                img_name = f"{name1}_{img_count}.png"  # Save images with index (e.g., name1_0.png)
                path = 'Training images'
                cv2.imwrite(os.path.join(path, img_name), frame)
                print("{} written!".format(img_name))
                img_count += 1  # Increment image count

        cam.release()  # Release the camera after capturing the required images
        cv2.destroyAllWindows()
        return render_template('image.html')
    else:
        return 'All is not well'

@app.route("/", methods=["GET", "POST"])
def recognize():
    if request.method == "POST":
        path = 'Training images'
        images = []
        classNames = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

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

        cap = cv2.VideoCapture(0)

        prev_ear = 0.0  # Store previous EAR value for comparison

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            # Convert the image to grayscale and detect faces
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

                # If the EAR is below a certain threshold, it suggests a photo or video, not a real person
                if ear < 0.25:
                    live = False
                else:
                    live = True

                # Checking the consistency of EAR over a few frames for better accuracy
                if abs(prev_ear - ear) < 0.05:  # If EAR value is stable, likely to be a photo
                    live = False
                prev_ear = ear

            if not live:
                cv2.putText(img, "Spoofing suspected. Blink not detected.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Punch your Attendance', img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if faceDis[matchIndex] < 0.50 and live:
                    name = classNames[matchIndex].upper()
                    markAttendance(name)
                    markData(name)
                    cap.release()  # Release the camera after successful attendance marking
                    cv2.destroyAllWindows()
                    return render_template('first.html')
                else:
                    name = 'Unknown'

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Punch your Attendance', img)
            if cv2.waitKey(1) & 0xFF == 27:
                breakf

        cap.release()
        cv2.destroyAllWindows()
        return render_template('first.html')
    else:
        return render_template('main.html')

# Remaining routes and login handling are unchanged.
@app.route('/login',methods = ['POST'])
def login():
    #print( request.headers )
    json_data = json.loads(request.data.decode())
    username = json_data['username']
    password = json_data['password']
    #print(username,password)
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
    #print('here')
    if 'username' in session:
        return session['username']
    return 'False'


@app.route('/how',methods=["GET","POST"])
def how():
    return render_template('form.html')
@app.route('/data',methods=["GET","POST"])
def data():
    '''user=request.form['username']
    pass1=request.form['pass']
    if user=="tech" and pass1=="tech@321" :
    '''
    if request.method=="POST":
        today=date.today()
        print(today)
        conn = sqlite3.connect('information.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        print ("Opened database successfully");
        cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance where Date=?",(today,))
        rows=cur.fetchall()
        print(rows)
        for line in cursor:

            data1=list(line)
        print ("Operation done successfully");
        conn.close()

        return render_template('form2.html',rows=rows)
    else:
        return render_template('form1.html')


            
@app.route('/whole',methods=["GET","POST"])
def whole():
    today=date.today()
    print(today)
    conn = sqlite3.connect('information.db')
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor() 
    print ("Opened database successfully");
    cursor = cur.execute("SELECT DISTINCT NAME,Time, Date from Attendance")
    rows=cur.fetchall()    
    return render_template('form3.html',rows=rows)


if __name__ == '__main__':
    app.run(debug=True)

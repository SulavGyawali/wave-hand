import cv2
import numpy as np
import face_recognition
import os
from threading import Thread
import time
import speech_recognition

known_face_encoding = []
known_face_names = []

def load_faces():
    try:
        file_list = os.listdir("faces")
    except FileNotFoundError:
        os.system("mkdir faces")
    with open("faces.txt", "w+") as f:
        for files in file_list:
            if files.startswith("."):
                continue
            f.write(files + "\n")

    with open("faces.txt", "r+") as f:
        images = f.readlines()
        for image in images:
            image = image.split("\n")[0]
            new_face = face_recognition.load_image_file(f"faces/{image}")
            new_encoding = face_recognition.face_encodings(new_face)[0]

            known_face_encoding.append(new_encoding)
            name = image.split(".")[0]
            known_face_names.append(name)


face_locations = []
face_encodings = []

hand_cascade = cv2.CascadeClassifier("palm.xml")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

name = None

affermative = ["yes","yeah","ofcourse","okay"]
negative = ["no","nah","sorry"]

cap = cv2.VideoCapture(0)

def take_speech():
    text = ""
    recognizer = speech_recognition.Recognizer()
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)

            text = recognizer.recognize_google(audio)
            text = text.lower()

            os.system(f"say Recognized {text}")
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
    
    return text

def register():
    os.system("say please state your name")
    name = input("Enter your name: ")
    os.system("say position yourself and say click to take a picture")
    response = take_speech()
    if "click" in response.split(" "):
        try:
            _, frame = cap.read()
            cv2.imwrite(f"faces/{name}.jpg",frame)
            load_faces()
            print("registered")
        except:
            pass
    elif response == "":
        os.system("say I couldnt understand you")


def couldnt_recognise():
    os.system("say Hey! I dont recognise you")
    time.sleep(0.5)
    os.system("say Would you like to register yourself?")
    response = take_speech()
    print(response)
    if any(word in response.split(" ") for word in affermative):
        register()
    elif any(word in response.split(" ") for word in negative):
        os.system("say Okay!")
    else:
        os.system("say sorry i dont understand you. Please try again!")
        couldnt_recognise()

recognise = Thread(target=couldnt_recognise)

def face(rgb):
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        if matches:
            face_distance = face_recognition.face_distance(
                known_face_encoding, face_encoding
            )
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                global name
                name = known_face_names[best_match_index]



def say_name(name):
    os.system(f"say hey {name} how are you?")


def video(frame, name, people,counter):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.5, 5)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 255, 0), -1)
        cv2.putText(
            frame,
            name,
            (x + 25, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
        )

    cv2.imshow("Video", frame)

    if type(hands) is not tuple:
        if name in people:
            Thread(target=say_name, args=(name,)).start()
            people.remove(name)
        elif name in known_face_names:
            pass
        else:
            if not recognise.is_alive():
                try:
                    recognise.start()
                except RuntimeError:
                    pass
            


def main():
    load_faces()
    counter = 0
    people = known_face_names.copy()

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            if counter % 30 == 0:
                try:
                    Thread(target=face, args=(rgb,)).start()
                except ValueError:
                    pass

            elif counter % 100 == 0:
                people = known_face_names.copy()
            counter += 1

        video(frame, name, people, counter)

        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    main()
cv2.destroyAllWindows()

import cv2
import numpy as np
import face_recognition
import os
from threading import Thread

sulav_face = face_recognition.load_image_file("faces/sulav.jpg")
sulav_encoding = face_recognition.face_encodings(sulav_face)[0]

kavi_image = face_recognition.load_image_file("faces/kavi.jpg")
kavi_encoding = face_recognition.face_encodings(kavi_image)[0]

suku_image = face_recognition.load_image_file("faces/suku.jpg")
suku_encoding = face_recognition.face_encodings(suku_image)[0]

known_face_encoding = [sulav_encoding, kavi_encoding, suku_encoding]
known_face_names = ["Sulav", "Kavi", "Suku"]

people = known_face_names.copy()

face_locations = []
face_encodings = []

hand_cascade = cv2.CascadeClassifier("palm.xml")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


name = None


def face(rgb):
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(
            known_face_encoding, face_encoding
        )
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            global name
            name = known_face_names[best_match_index]


def say_name(name):
    os.system(f"say hey {name} how are you?")


def video(frame, name):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

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


def main():
    cap = cv2.VideoCapture(0)
    counter = 0
    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            if counter % 50 == 0:
                try:
                    Thread(target=face, args=(rgb,)).start()
                except ValueError:
                    pass
            counter += 1

        video(frame, name)

        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    main()
cv2.destroyAllWindows()

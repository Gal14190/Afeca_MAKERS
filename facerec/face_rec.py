import face_recognition
import os
import cv2

KNOWN_FACES_DIR      = "members"
UNKNOWN_FACES_DIR    = "unknown_face"
TOLERANCE       = 0.1
FRAME_THICKNESS = 3
FONT_THICKNESS  = 2
MODEL = "cnn"

print("loading members")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    print(f"@@@ {name}")
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(f"%%% {KNOWN_FACES_DIR}/{name}/{filename}")

        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
        except:
            print("")
# SHOW WINDOW
    #cv2.imshow(name, image)
    #cv2.waitKey(0)
    #cv2.destroyWindow(name)

print("processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    print("check#0")
    locations = face_recognition.face_locations(image, model=MODEL)
    print("check#10")
    encodings = face_recognition.face_encodings(image, locations)
    print("check#01")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print("check#1")
    for face_encoding, face_location in zip(encoding, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        print("check#2")
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            cv2.Text(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHY_SIMPLEX)


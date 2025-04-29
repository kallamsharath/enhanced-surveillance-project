import cv2 
import face_recognition as fr
from twilio.rest import Client

# Twilio
account_sid = 'your_account_sid'  # Private data (original removed)
auth_token = 'your_auth_token'    # Private data (original removed)
twilio_number = 'your_twilio_number'  # Private data (original removed)
to_number = 'recipient_number'        # Private data (original removed)


def send_message():
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body="ALERT Unknown person detected in your office room!",
        from_=twilio_number, 
        to=to_number
    )
    print(f"[ALERT] Message sent! SID: {message.sid}")

# Load and encode known faces
known_face_encoding = []
known_face_names = []

known_p1 = fr.load_image_file("images/known/person1.jpg")
known_p2 = fr.load_image_file("images/known/person2.jpg")
known_p3 = fr.load_image_file("images/known/person3.jpg")

known_face_encoding.append(fr.face_encodings(known_p1)[0])
known_face_encoding.append(fr.face_encodings(known_p2)[0])
known_face_encoding.append(fr.face_encodings(known_p3)[0])

known_face_names.extend(["Sharath", "Sharath", "Sharath"])

# Start webcam
video_cap = cv2.VideoCapture(0)
alert_sent = False  # Flag to avoid sending multiple alerts

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = fr.compare_faces(known_face_encoding, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            if not alert_sent:
                send_message()
                alert_sent = True  # Send alert only once

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()

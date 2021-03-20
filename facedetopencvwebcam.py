# https://medium.com/analytics-vidhya/real-time-webcam-face-detection-system-using-opencv-in-python-windows-and-macos-86c31fddd2bc

# import libraries
import cv2
import face_recognition

# reference your system's webcam
video_capture = cv2.VideoCapture(0)

# Initialize the required variables. These variables will be populated later on in the code
face_locations = []

'''
We divide our video (real-time) into different frames. In each frame, 
we detect the location of the face using the APIs which we have imported 
above. For each face detected, we locate the coordinates and draw a 
rectangle around it and release the video to the viewer.
'''
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # convert the image from RGB color (which OpenCV uses) to RGB color
    # (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # display the results
    for top, right, bottom, left in face_locations:
        # draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
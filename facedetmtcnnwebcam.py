import cv2
from mtcnn import MTCNN

ksize = (101, 101)
font = cv2.FONT_HERSHEY_SIMPLEX

def find_face_MTCNN(color, result_list):
    for result in result_list:
        x, y, w, h = result['box']
        roi = color[y:y+h, x:x+w]
        cv2.rectangle(color,
                      (x, y), (x+w, y+h),
                      (0, 155, 255),
                      5)
        detectedFace = cv2.GaussianBlur(roi, ksize, 0)
        color[y:y+h, x:x+w] = detectedFace
    return color


video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = MTCNN()

while True:
    _, color = video_capture.read()
    faces = detector.detect_faces(color)
    detectFaceMTCNN = find_face_MTCNN(color, faces)
    cv2.imshow('Video', detectFaceMTCNN)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
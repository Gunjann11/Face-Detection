import cv2 // opencv uses bgr instead of rgb 

#It opens webcam feed and prepares each frame (flipped + grayscale) for face detection 

face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video_cap = cv2.VideoCapture(0)

while True:
    ret, video = video_cap.read()
    video= cv2.flip(video, 1)
    col = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    
#It finds faces using the Haar cascade model and highlights them with bounding boxes on the video

    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 0), 2)


#It shows the processed video feed, waits for the key “a” to exit the loop, and then properly releases the camera and closes the display window.
    
    cv2.imshow("video_live", video)

    if cv2.waitKey(10) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()

# Press a to close the camera

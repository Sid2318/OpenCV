# for image

# import cv2 as cv

# img = cv.imread('image.png')
# img = cv.resize(img, (400,350))
# # cv.imshow("Person",img)

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # cv.imshow("Gray",gray)

# haar_cascade = cv.CascadeClassifier('haar_face.xml')

# # if haar_cascade.empty():
# #     print("Error: Haar cascade file not loaded properly.")
# # else:
# #     print("Haar cascade loaded successfully.")

# faces_react = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors = 2)
# print(faces_react)                        # [[102  88 176 176]]
# print(f'Number of faces found = {len(faces_react)}')

# for (x,y,w,h) in faces_react:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness =2)

# cv.imshow("Detected",img)

# cv.waitKey(0)





#for video
import cv2 as cv
import datetime  # Import datetime module

cap = cv.VideoCapture(0)

# Load Haar Cascade only once (instead of inside the loop)
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    # Resize frame
    img = cv.resize(frame, (400, 350))

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Face detection
    faces_react = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    print(f'Number of faces found = {len(faces_react)}')

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces_react:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # 🕒 Add timestamp to the frame
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv.putText(img, timestamp, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the frame
    cv.imshow("Detected", img)

    # Exit on 'q' key press
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
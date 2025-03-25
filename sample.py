import cv2
import imutils
import dlib
from scipy.spatial import distance
from imutils import face_utils
import winsound  # Import winsound to play a beep sound

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize constants
thresh = 0.25  # Threshold for EAR to detect drowsiness
frame_check = 20  # Consecutive frames to confirm drowsiness
detect = dlib.get_frontal_face_detector()  # Face detector
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Load facial landmark predictor

# Eye landmarks indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)  # Open webcam
flag = 0  # Counter for consecutive frames with low EAR

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)  # Resize frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    subjects = detect(gray, 0)  # Detect faces

    for subject in subjects:
        shape = predict(gray, subject)  # Get face landmarks
        shape = face_utils.shape_to_np(shape)  # Convert to numpy array

        leftEye = shape[lStart:lEnd]  # Get left eye landmarks
        rightEye = shape[rStart:rEnd]  # Get right eye landmarks

        leftEAR = eye_aspect_ratio(leftEye)  # Calculate EAR for left eye
        rightEAR = eye_aspect_ratio(rightEye)  # Calculate EAR for right eye

        ear = (leftEAR + rightEAR) / 2.0  # Average EAR

        leftEyeHull = cv2.convexHull(leftEye)  # Convex hull for left eye
        rightEyeHull = cv2.convexHull(rightEye)  # Convex hull for right eye

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # Draw left eye contour
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # Draw right eye contour

        # Check if EAR is below the threshold (indicating drowsiness)
        if ear < thresh:
            flag += 1  # Increment flag if EAR is low
            print(flag)
            if flag >= frame_check:
                # Alert the user with a message and beep sound
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                winsound.Beep(1000, 500)  # Play a beep sound (1000 Hz for 500 ms)
        else:
            flag = 0  # Reset flag if EAR is above threshold

    cv2.imshow("Frame", frame)  # Display the frame

    key = cv2.waitKey(1) & 0xFF  # Wait for key press
    if key == ord("q"):  # Exit loop if "q" is pressed
        break

cv2.destroyAllWindows()  # Close all OpenCV windows
cap.release()  # Release the video capture object

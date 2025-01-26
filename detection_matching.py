import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# Helper function to calculate face descriptor
def get_face_descriptor(gray, face):
    landmarks = predictor(gray, face)
    points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
    return points

# Load target face image and calculate descriptor
target_image_path = "target_face.jpeg"
target_image = cv2.imread(target_image_path)

if target_image is None:
    print(f"Error: Unable to load target image from {target_image_path}.")
    exit()

target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
target_faces = detector(target_gray)

if len(target_faces) == 0:
    print("No face detected in the target image.")
    exit()

try:
    target_face_descriptor = get_face_descriptor(target_gray, target_faces[0])
except Exception as e:
    print(f"Error generating face descriptor for target image: {e}")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        try:
            # Calculate descriptor for the current face
            face_descriptor = get_face_descriptor(gray, face)
            
            # Compare descriptors using Euclidean distance
            similarity = distance.euclidean(target_face_descriptor.flatten(), face_descriptor.flatten())

            # Display similarity score
            if similarity < 50:  # Threshold for matching (adjust based on testing)
                match_text = "Matched"
                color = (0, 255, 0)
            else:
                match_text = "Not Matched"
                color = (0, 0, 255)
            
            cv2.putText(frame, f"{match_text}: {similarity:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception as e:
            cv2.putText(frame, "Error processing face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Error processing face: {e}")

    cv2.imshow('Face Matching', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

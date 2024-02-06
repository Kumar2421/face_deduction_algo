import cv2
import os
import numpy as np
import imutils
import pickle
import joblib

# Load the serialized face detector from disk
print("Loading Face Detector...")
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

## Load serialized face embedding model
model_path = r'E:\face\face-deduction-algo-main\face-recognition-using-deep-learning-master\face_detection_model\openface_nn4.small2.v1.t7'
print(f"Loading Face Recognizer from {model_path}...")
embedder = cv2.dnn.readNetFromTorch(model_path)

# Load the embeddings and the label encoder from the dataset
data = pickle.loads(open('output/embeddings.pickle', "rb").read())
embeddings = data["embeddings"]
# Load the label encoder using joblib
le = joblib.load("output/le.joblib")

# Initialize video capture from the default camera (usually 0)
cap = cv2.VideoCapture(0)

# Create a flag to indicate if an image should be captured
capture_image = False

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Resize the frame for faster processing (adjust width as needed)
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Create a blob from the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

    # Pass the blob through the face detector to detect faces
    detector.setInput(blob)
    detections = detector.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # Create a blob for the face ROI, then pass the blob through the face embedding model
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Calculate L2 distances between the embeddings and the known embeddings
            distances = np.linalg.norm(embeddings - vec, axis=1)
            min_distance = np.min(distances)

            # Define a threshold for face recognition (adjust as needed)
            threshold = 0.5

            # If the minimum distance is below the threshold, consider it as a known face
            if min_distance < threshold:
                # Find the label associated with the recognized face
                idx = np.argmin(distances)
                if idx < len(le.classes_):
                    label = le.inverse_transform([idx])[0]  # Decode the label using the label encoder
                else:
                    label = "Unknown"  # Assign to an "Unknown" label
            else:
                label = "Unknown"  # Assign to an "Unknown" label

            y = startY - 10 if startY - 10 > 10 else startY + 10
            text = f"User: {label}"  # Define the text to display
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Show the frame with detections
    cv2.imshow("Face Recognition", frame)

    # Check if the capture_image flag is True and save the image
    if capture_image:
        cv2.imwrite("captured_image.jpg", frame)
        print("Image captured!")

        # Reset the flag to False
        capture_image = False

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):  # Press 'c' to capture an image
        capture_image = True

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

import logging
from imutils import paths, resize  # Import the 'resize' function from 'imutils'
import numpy as np
import pickle
import cv2
import os

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_info(message):
    """
    Logs an info message to the app.log file.
    Args:
        message (str): The info message to log.
    """
    logging.info(message)

def log_error(message):
    """
    Logs an error message to the app.log file.
    Args:
        message (str): The error message to log.
    """
    logging.error(message)

# Load serialized face detector
log_info("Loading Face Detector...")
protoPath = "E:\\face\\face-deduction-algo-main\\face_detection_model\\deploy.prototxt"
modelPath = "E:\\face\\face-deduction-algo-main\\face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load serialized face embedding model
model_path = r'E:\\face\\face-deduction-algo-main\\face_detection_model\\openface_nn4.small2.v1.t7'
log_info(f"Loading Face Recognizer from {model_path}...")
embedder = cv2.dnn.readNetFromTorch(model_path)

# Grab the paths to the input images in our dataset
log_info("Quantifying Faces...")
imagePaths = list(paths.list_images("E:\\face\\face-deduction-algo-main\\dataset"))

# Initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []

# Initialize the total number of faces processed
total = 0

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path
    if (i % 50 == 0):
        log_info("Processing image {}/{}".format(i, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # Load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(imagePath)
    image = resize(image, width=600)  # Use the 'resize' function
    (h, w) = image.shape[:2]

    # Construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Ensure at least one face was found
    if len(detections) > 0:
        # We're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Ensure the face width and height are sufficiently large
            if fW >= 20 and fH >= 20:
                # Construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # Add the name of the person + corresponding face embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

# Dump the facial embeddings + names to disk
log_info("[INFO] Serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

# Close the log file
logging.shutdown()

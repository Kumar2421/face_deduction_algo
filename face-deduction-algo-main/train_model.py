import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import joblib  # Import joblib
import numpy as np

# Configure the logging module to use a different log file
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

# Load the face embeddings or create a new one if not found
try:
    with open("E:\\face\\face-deduction-algo-main\\output\\embeddings.pickle", "rb") as embeddings_file:
        data = pickle.load(embeddings_file)
except FileNotFoundError:
    log_error("[ERROR] embeddings.pickle file not found. Creating a new one.")
    data = {"embeddings": [], "names": []}

# Ensure that data["embeddings"] is not empty
if not data["embeddings"]:
    log_error("[ERROR] Face embeddings array is empty.")
    exit()

# Reshape the embeddings to a 2D array
embeddings_2d = np.array(data["embeddings"]).reshape(len(data["embeddings"]), -1)

# Your existing code to encode labels
log_info("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# Check the number of unique classes
unique_classes = np.unique(labels)
if len(unique_classes) < 2:
    log_error("[ERROR] Number of unique classes is less than 2. Unable to train the model.")
    exit()

# Print the number of unique classes
log_info(f"[INFO] Number of unique classes: {len(unique_classes)}")

# Your existing code to train the model
log_info("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(embeddings_2d, labels)

# Your existing code to save the model and label encoder to disk
log_info("[INFO] saving recognizer model to disk...")
joblib.dump(recognizer, "output/recognizer.joblib")

log_info("[INFO] saving label encoder to disk...")
joblib.dump(le, "output/le.joblib")

log_info("[INFO] Face recognition model training completed from script third.")

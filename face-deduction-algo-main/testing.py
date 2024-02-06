import cv2
import os
import time
import psutil
import shutil
import logging
import subprocess

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

admin_password = "admin123"

# Initialize camera with error handling
def initialize_camera():
    try:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        return cap
    except Exception as e:
        log_error(f"Error initializing camera: {str(e)}")
        return None

# Create or access output directory with error handling
def create_output_directory(file_name):
    try:
        output_dir = os.path.join("E:\\face\\face-deduction-algo-main\\dataset", file_name)

        if os.path.exists(output_dir):
            admin_password_attempt = input("Enter the admin password to continue capturing images: ")
            if admin_password_attempt != admin_password:
                logging.error("Invalid admin password. Access denied.")
                log_info(f"Access denied for user '{file_name}'")
                exit()
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        log_info(f"Created/accessed output directory: {output_dir}")
        return output_dir
    except Exception as e:
        log_error(f"Error creating/accessing output directory: {str(e)}")
        return None

# Capture frames with error handling
def capture_frames(cap, output_dir, video_duration):
    try:
        # Define video writer to save the captured video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(os.path.join(output_dir, "captured_video.avi"), fourcc, 20.0, (640, 480))

        # Counter for captured images
        capture_count = 0

        video_start_time = time.time()  # Start measuring video capture time

        while (time.time() - video_start_time) < video_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            # Display the frame in a window
            cv2.imshow("Capture Video", frame)

            # Write the frame to the video file
            video_out.write(frame)

            # Save the captured image in the specified location
            frame_file_name = os.path.join(output_dir, f"frame_{capture_count:04d}.jpg")
            cv2.imwrite(frame_file_name, frame)
            capture_count += 1
            print(f"Captured {capture_count} frames.")

            # Check for the 'q' key to stop the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_out.release()
        cv2.destroyAllWindows()

        log_info(f"Captured {capture_count} frames.")
        return capture_count
    except Exception as e:
        log_error(f"Error capturing frames: {str(e)}")
        return None

def main():
    try:
        # Input the person's name
        file_name = input("Enter the name of the person: ")
        log_info(f"User entered name: {file_name}")

        # Set the video capture duration (in seconds)
        video_duration = 30  # Adjust as needed

        cap = initialize_camera()
        if cap is None:
            logging.error("Camera initialization failed. Exiting.")
            quit()  # Exit the script if camera initialization fails

        output_dir = create_output_directory(file_name)
        if output_dir is None:
            logging.error("Output directory creation failed. Exiting.")
            quit()  # Exit the script if output directory creation fails

        start_time = time.time()  # Start measuring execution time
        capture_count = capture_frames(cap, output_dir, video_duration)
        if capture_count is None:
            logging.error("Frame capture failed. Exiting.")
            quit()  # Exit the script if frame capture fails
        end_time = time.time()  # Stop measuring execution time
        execution_time = end_time - start_time

        # Get CPU usage
        cpu_usage = psutil.cpu_percent()

        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"CPU Usage: {cpu_usage}%")
        print(f"{capture_count} frames captured and saved in the directory '{output_dir}'.")

        log_info(f"Execution time: {execution_time:.2f} seconds")
        log_info(f"CPU Usage: {cpu_usage}%")
        log_info(f"{capture_count} frames captured and saved in the directory '{output_dir}'.")
    except Exception as e:
        log_error(f"An error occurred: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    # Security features
    while True:
        user_choice = input("Enter 'view', 'deletefolder', or 'quit': ").lower()

        if user_choice == "view":
            # View files in the directory
            files = os.listdir(output_dir)
            if files:
                print("Files in the directory:")
                for file in files:
                    print(file)
            else:
                print("No files in the directory.")
        
        elif user_choice == "deletefolder":
            # Delete the entire folder (password-protected)
            password = input("Enter the delete password: ")
            if password == admin_password:
                try:
                    shutil.rmtree(output_dir)
                    print(f"Folder '{output_dir}' and its contents deleted.")
                    log_info(f"Folder '{output_dir}' and its contents deleted.")
                except Exception as e:
                    log_error(f"Error deleting the folder: {str(e)}")
            else:
                logging.error("Invalid delete password. Access denied.")
                log_info(f"Invalid delete password entered.")
        elif user_choice == "quit":
            print("Quitting the first script. Running the second script...")
            break  # Break out of the loop and proceed to run the second script
        else:
            print("Invalid input. Please enter 'view', 'deletefolder', or 'quit'.")

    # Run the second script after quitting the first script
    print("Running the second script...")
    subprocess.run(["python", "extract_embeddings.py"])
    print("Second script completed. Running the third script...")
    subprocess.run(["python", "train_model.py"])

if __name__ == "__main__":
    main()

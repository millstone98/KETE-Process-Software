import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import os
import time
import requests

API_URL = "https://657ca02c853beeefdb99bbf0.mockapi.io/face_count"

def send_data_to_api(data):
    try:
        response = requests.post(API_URL, json=data)
        response.raise_for_status()
        print("Data sent successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data: {e}")

def count_faces(output_folder='/home/raphael.bernhardsgru/Pictures', capture_interval=5):
    # Ensure the output folder exists
    try:
        os.mkdir(output_folder)
    except OSError:
        pass  # If the directory already exists, ignore the exception

    # Initialize PiCamera and PiRGBArray
    with PiCamera() as camera:
        # Set camera resolution
        camera.resolution = (1280, 1024)

        # Set ISO to adjust brightness (100 to 800, higher values for brighter images)
        camera.iso = 200

        # Set shutter speed (exposure time) in microseconds
        camera.shutter_speed = 25000

        # Wait for the camera to warm up
        time.sleep(1)

        while True:
            # Initialize the PiRGBArray
            raw_capture = PiRGBArray(camera)

            # Capture a single frame
            camera.capture(raw_capture, format="bgr")
            image = raw_capture.array

            # Rotate the image (change the angle as needed)
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)

            # Load the local Haar Cascade classifier for face detection
            face_cascade_path = "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(face_cascade_path)

            # Convert the rotated image to grayscale for face detection
            gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

            # Perform face detection
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

            # Draw green rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(rotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the rotated image with rectangles around faces in the specified folder
            output_path = output_folder + '/image_with_faces.jpg'
            with open(output_path, 'wb') as output_file:
                output_file.write(cv2.imencode('.jpg', rotated_image)[1])

            # Save the number of faces to the variable 'faces'
            faces_count = len(faces)

            # Print the number of faces found
            print(f"Number of faces: {faces_count}")

            # Send data to the API
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            data_to_send = {"faces_count": faces_count, "timestamp": current_time}
            send_data_to_api(data_to_send)

            # Wait for the specified capture interval
            time.sleep(capture_interval)

if __name__ == "__main__":
    count_faces()

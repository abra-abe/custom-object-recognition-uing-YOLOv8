import cv2
import os
from roboflow import Roboflow

# Initialize Roboflow
RoboflowAPIKey = os.getenv('api_key')
rf = Roboflow(api_key=RoboflowAPIKey)
project = rf.workspace().project("fish-plastic-detection")
model = project.version(5).model


# def imageDetection():
#     # Perform inference
#     response = model.predict("plasticBottles.jpg", confidence=30, overlap=30).json()

#     # Load image with OpenCV
#     image = cv2.imread("plasticBottles.jpg")

#     # resizing the image
#     img = cv2.resize(image, (224, 224))

#     # Iterate over predictions and draw bounding boxes
#     for prediction in response["predictions"]:
#         label = prediction["class"]
#         x = int(prediction["x"])
#         y = int(prediction["y"])
#         height = int(prediction["height"])
#         width = int(prediction["width"])
#         confidence = float(prediction["confidence"])

#         # Calculate bounding box coordinates
#         ymin = (y - 70)
#         xmin = (x - 70)
#         ymax = (y + 60)
#         xmax = (x + 80)

#         # Draw bounding box rectangle
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

#         # counting the number of objects detected
#         c = label.count('PlasticBottle')
#         print(c)

#         # Display label and confidence
#         label_text = f"{label}: {confidence:.2f} {c}"
#         cv2.putText(image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)


#     # Display the image with bounding boxes
#     cv2.imshow("Image with Predictions", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# real time video detection
def videoDetection():
    video_capture = cv2.VideoCapture("/dev/video2")

    count = 0
    while True:
        # Read each frame from the video feed
        ret, frame = video_capture.read()
        # reducing the number of frames to be processed
        count += 1
        if count % 20 != 0:
            continue

        # Perform inference on the frame
        response = model.predict(frame, confidence=30, overlap=30).json()

        # Iterate over predictions and draw bounding boxes
        for prediction in response["predictions"]:
            label = prediction["class"]
            x = int(prediction["x"])
            y = int(prediction["y"])
            height = int(prediction["height"])
            width = int(prediction["width"])
            confidence = float(prediction["confidence"])

            # Calculate bounding box coordinates
            xmin = x - width
            xmax = x + width
            ymin = y - height
            ymax = y + height

            # Draw bounding box rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display label and confidence
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # counting the number of objects detected
        # plastics = label.count('Plastic')
        # print(plastics)
        # fish = label.count('fish')
        # print(fish)

        # Display the frame with bounding boxes
        cv2.imshow("Frame with Predictions", frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# option = input("1 or 2")
# print("\n")
# if option == '1':
#     imageDetection()
# elif option == '2':
#     videoDetection()
# else:
#     print('invalid option')

videoDetection()

print('TNE END')
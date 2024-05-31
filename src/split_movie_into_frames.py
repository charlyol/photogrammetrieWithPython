import cv2
import os

video_path = os.path.abspath('/home/olinger/PycharmProjects/photogrammetrieWithPython/data/movie')
output_folder = os.path.abspath('/home/olinger/PycharmProjects/photogrammetrieWithPython/data/frames')


def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the video
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video is opened correctly
    if not video_capture.isOpened():
        print(f"Error opening video {video_path}")
        return

    frame_count = 0
    while True:
        # Read a frame
        ret, frame = video_capture.read()

        # If the frame is not read correctly, exit the loop
        if not ret:
            break

        # Construct the frame file name
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the resources
    video_capture.release()
    print(f"Extraction completed. {frame_count} frames extracted to the folder {output_folder}")


# Example usage
extract_frames(video_path, output_folder)

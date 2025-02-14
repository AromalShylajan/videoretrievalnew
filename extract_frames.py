import cv2
from PIL import Image

def extract(video):
    # The frame images will be stored in video_frames
    video_frames = []

    # Open the video file
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)

    current_frame = 0
    while capture.isOpened():
    # Read the current frame
        ret, frame = capture.read()

        # Convert it to a PIL image (required for CLIP) and store it
        if ret == True:
            video_frames.append(Image.fromarray(frame[:, :, ::-1]))
        else:
            break

        # Skip N frames
        current_frame += 120
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # Print some statistics
    print(f"Frames extracted: {len(video_frames)}")
    return video_frames
import numpy as np
import cv2
import os

image = cv2.imread('lena-1.png')

def print_image_information(image):
    height,width,channels = image.shape
    print("Height:",height)
    print("Width:",width)
    print("Channels:",channels)
    print("Size:",image.size)
    print("Data type:",image.dtype)



if image is None:
    print("Error no image found")

else:
    print_image_information(image)

filsti = "solutions/camera_outputs.txt"

os.makedirs(os.path.dirname(filsti), exist_ok=True)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error no camera opened")

else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with open(filsti,"w") as f:
        f.write(f"fps: {int(fps)}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")

        print(f"Camera info saved to {filsti}")

cap.release()
cv2.destroyAllWindows()
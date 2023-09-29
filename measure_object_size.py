import cv2
from object_detector import *
import numpy as np
import socket
import io
from PIL import Image
from PIL import ImageDraw
import tqdm
import os

#creation du buffer
SEPARATOR = "<@>"
BUFFER_SIZE = 4096

host = "192.168.1.117"

port = 7873
s = socket.socket()
print(f"[+] Connecting to {host}:{port}")

s.connect((host, port))

while True:
    # Load Aruco detector
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
    received = s.recv(4096)
    if not received:
        continue
    else:
        a = received.decode()
        filename,filesize = a.split(SEPARATOR)
        print(filename)
        filename = os.path.basename(filename)
        filesize = int(filesize)
        progress = tqdm.tqdm(range(filesize),f"Receiving{filename}",unit="B", unit_scale=True, unit_divisor=1024)
        with open(filename, "wb") as f:
            while True:
                bytesr = s.recv(4096)

                if not bytesr:
                    break
                f.write(bytesr)
                progress.update(len(bytesr))
        s.close()
        # Load Object Detector
        detector = HomogeneousBgDetector()


        # Load Image
        # Get Aruco marker
        filename = np.array(filename)
        corners, _, _ = cv2.aruco.detectMarkers(filename, aruco_dict, parameters=parameters)

        # Draw polygon around the marker
        #int_corners = np.uint8(corners)
        int_corners = np.array(corners)
        cv2.polylines(filename, int_corners, True, (0, 255, 0), 5)

        # Aruco Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20

        contours = detector.detect_objects(filename)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.circle(filename, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(filename, [box], True, (255, 0, 0), 2)
            cv2.putText(filename, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(filename, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)



    cv2.imshow("Image", filename)
    cv2.waitKey(0)
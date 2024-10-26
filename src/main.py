from pathlib import Path

import cv2
import numpy as np

# from utils.logger import get_logger
from loguru import logger

from aruco.aruco import ArucoMapService
from utils.file_handler import LocationConfigHandler

if __name__ == "__main__":

    aruco_map_service = ArucoMapService(aruco_dict_id=cv2.aruco.DICT_4X4_100)

    logger.info("Hello, World!")
    location_folder = Path("locations")
    capitols_file = location_folder / "capitols.yaml"

    capitols = LocationConfigHandler(capitols_file).parse_locations()

    image_folder = Path("images")
    image_file = image_folder / "map_with_markers_and_input2.png"
    image = cv2.imread(image_file)

    if image is None:
        raise FileNotFoundError(f"Could not read image {image_file}")

    frame = aruco_map_service.extract_frame(image=image, draw_markers=True)

    cv2.imshow("Extracted frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # detect red dot in the frame image
    # filter all non red pixels
    # convert to grayscale
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the two ranges and combine them
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixels_only = cv2.bitwise_and(frame, frame, mask=red_mask)
    cv2.imshow("Threshold", red_pixels_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_image = cv2.cvtColor(red_pixels_only, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and improve detection
    th, threshed = cv2.threshold(
        gray_image, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    logger.info(f"Found {len(cnts)} contours")
    logger.info(f"Contours: {cnts}")

    # Draw the contours on the original image
    for cnt in cnts:
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    # draw line from the center of the image to the center of the circle
    for cnt in cnts:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.line(
            frame, (frame.shape[1] // 2, frame.shape[0] // 2), center, (0, 255, 0), 2
        )

    # Display the output image with detected circles
    cv2.imshow("Red Circles Detected", frame)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

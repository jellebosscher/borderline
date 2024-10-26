from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from utils.image import augment_image

image_folder = Path("images")

image = cv2.imread(image_folder / "map_with_markers.png")

logger.info(f"Type of image: {type(image)}")


def detect_aruco_markers(image, aruco_dict_id=cv2.aruco.DICT_4X4_100):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    return aruco_detector.detectMarkers(image)


def retrieve_orthogonal_rectangle_between_markers(
    image: np.ndarray, detected_markers_dict: dict
):
    """
    Retrieve the transformed rectangle between the detected markers in the image.
    The rectangle is defined by the inner corners of four detected markers.

    cv2 aruco coordinates are ordered as:
        0: top left corner
        1: top right
        2: bottom right
        3: bottom left

    Parameters:
        image: The image containing the detected markers.
        detected_markers: The four detected markers that form the corners of the rectangle.
            top-left: bottom-right of marker with ID 1
            top-right: bottom-left of marker with ID 2
            bottom-right: top-left of marker with ID 3
            bottom-left: top-right of marker with ID 4

    Returns:
        Image containing only the rectangle defined by the detected markers.
    """
    if len(detected_markers_dict) < 4:
        raise ValueError("At least four markers are required to define the rectangle.")

    # Extract the necessary corners of the markers in the expected order
    # aruco marker corners are ordered as
    top_left = detected_markers_dict[1][2]
    top_right = detected_markers_dict[2][3]
    bottom_right = detected_markers_dict[3][0]
    bottom_left = detected_markers_dict[4][1]

    src_points = np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
    )

    # Define the destination points for the orthogonal rectangle
    image_height, image_width = image.shape[:2]
    dst_points = np.array(
        [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (image_width, image_height))

    return warped_image


if __name__ == "__main__":

    images = augment_image(image)
    for image in images:
        cv2.imshow("Image after aug, before detection.", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        corners, ids, _ = detect_aruco_markers(image)
        if ids is not None:
            logger.info(f"Detected {len(ids)} markers")

        marker_dict = {
            id[0]: marker_corners[0] for id, marker_corners in zip(ids, corners)
        }

        image = retrieve_orthogonal_rectangle_between_markers(image, marker_dict)

        cv2.imshow("Image after transformation", image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

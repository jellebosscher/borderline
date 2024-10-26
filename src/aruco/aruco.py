import cv2
import numpy as np


class ArucoMapService:

    def __init__(self, aruco_dict_id=cv2.aruco.DICT_4X4_100):
        self.aruco_dict_id = aruco_dict_id

    @staticmethod
    def _draw_markers(image, corners, ids) -> None:
        """
        Draw the detected markers on a copy of the image.

        Parameters:
            image: The image containing the detected markers.
            corners: The corners of the detected markers.
            ids: The IDs of the detected markers.

        Returns:
            Image with the detected markers drawn on it.
        """
        image = image.copy()
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow("Image after detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_aruco_markers(self, image):
        """
        detected_markers: The four detected markers that form the corners of the rectangle.
            top-left: bottom-right of marker with ID 1
            top-right: bottom-left of marker with ID 2
            bottom-right: top-left of marker with ID 3
            bottom-left: top-right of marker with ID 4
        """
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        return aruco_detector.detectMarkers(image)

    def extract_frame(
        self, image: np.ndarray, draw_markers: bool = False
    ) -> np.ndarray:
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
            draw_markers: Whether to draw the detected markers on the image.


        Returns:
            Image containing only the rectangle defined by the detected markers.
        """

        corners, ids, _ = self.detect_aruco_markers(image)

        if len(ids) != 4:
            raise ValueError(
                "Exactly four markers are required to define the rectangle."
            )

        if draw_markers:
            self._draw_markers(image, corners, ids)

        detected_markers_dict = {
            id[0]: marker_corners[0] for id, marker_corners in zip(ids, corners)
        }

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

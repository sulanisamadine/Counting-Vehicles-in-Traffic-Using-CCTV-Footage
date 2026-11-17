import cv2
import numpy as np

class DIPProcessor:
    """
    Digital Image Processing Techniques
    CS31092 - Digital Image Processing | Group 40
    Techniques: Noise Removal, Sharpening, CLAHE, Brightness Norm, Morphological Ops
    """

    def apply_noise_removal(self, frame):
        gaussian = cv2.GaussianBlur(frame, (5, 5), 0)
        result   = cv2.medianBlur(gaussian, 3)
        return result

    def apply_sharpening(self, frame):
        kernel   = np.array([[0, -1,  0],
                             [-1,  5, -1],
                             [0, -1,  0]])
        return cv2.filter2D(frame, -1, kernel)

    def apply_contrast_enhancement(self, frame):
        lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l        = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def apply_brightness_normalization(self, frame):
        hsv          = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_morphological_operations(self, frame):
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def preprocess(self, frame):
        """Apply all 5 DIP techniques in sequence"""
        frame = self.apply_noise_removal(frame)
        frame = self.apply_sharpening(frame)
        frame = self.apply_contrast_enhancement(frame)
        frame = self.apply_brightness_normalization(frame)
        frame = self.apply_morphological_operations(frame)
        return frame
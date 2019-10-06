import cv2

class AugmentationTool:
    @staticmethod
    def RotateImage(image, angle):
        if angle == 0.0:
            rotated_mat = image.copy()
            return rotated_mat
        elif angle == 90.0:
            rotated_mat = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return rotated_mat
        elif angle == 180.0:
            rotated_mat = cv2.rotate(image, cv2.ROTATE_180)
            return rotated_mat
        elif angle == 270.0:
            rotated_mat = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            return rotated_mat
        else:
            height, width = image.shape[:2]
            image_center = (width/2, height/2)

            rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

            abs_cos = abs(rotation_mat[0,0])
            abs_sin = abs(rotation_mat[0,1])

            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_mat[0, 2] += bound_w/2 - image_center[0]
            rotation_mat[1, 2] += bound_h/2 - image_center[1]

            rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
            return rotated_mat

    @staticmethod
    def FlipImageHorizontally(image):
         flipped_image = cv2.flip(image, 0)
         return flipped_image
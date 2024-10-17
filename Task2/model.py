import cv2
import numpy
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection


class ImageMatching:
    # Class model and processor
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

    def __init__(self, img_path_1: str, img_path_2: str, count: int = 0):
        self.image_1 = Image.open(img_path_1)
        self.image_2 = Image.open(img_path_2)
        self.matches_count = count
        self.keypoint_1 = None
        self.keypoint_2 = None
        self.matches = None

    def _prepare_image(self):
        images = [self.image_1, self.image_2]

        inputs = ImageMatching.processor(images, return_tensors="pt").to(ImageMatching.model.device, ImageMatching.model.dtype)
        outputs = ImageMatching.model(**inputs)

        image_sizes = [(image.size[1], image.size[0]) for image in images]
        outputs = ImageMatching.processor.post_process_keypoint_detection(outputs, image_sizes)

        # Keypoint saving
        self.keypoint_1 = outputs[0]["keypoints"]
        self.keypoint_2 = outputs[1]["keypoints"]

        # Keypoint matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        self.matches = bf.match(outputs[0]["descriptors"].detach().numpy(), outputs[1]["descriptors"].detach().numpy())

    def compare_image(self):
        # Convert to cv2 image
        self._prepare_image()

        img1 = numpy.array(self.image_1)[:, :, ::-1].copy()
        img2 = numpy.array(self.image_2)[:, :, ::-1].copy()

        # Concat to images
        combined_image = numpy.hstack((img1, img2))

        # Draw lines
        if self.matches_count == 0:
            for match in self.matches:  # Отображаем только первые 10 совпадений
                pt1 = (int(self.keypoint_1[match.queryIdx][0]), int(self.keypoint_1[match.queryIdx][1]))
                pt2 = (int(self.keypoint_2[match.trainIdx][0] + img1.shape[1]), int(self.keypoint_2[match.trainIdx][1]))

                cv2.line(combined_image, pt1, pt2, (0, 255, 0), 2)  # Зеленая линия
                cv2.circle(combined_image, pt1, 5, (0, 0, 255), -1)  # Красная точка на первом изображении
                cv2.circle(combined_image, pt2, 5, (0, 0, 255), -1)  # Красная точка на втором изображении
        else:
            for match in self.matches[:self.matches_count]:  # Отображаем только первые 10 совпадений
                pt1 = (int(self.keypoint_1[match.queryIdx][0]), int(self.keypoint_1[match.queryIdx][1]))
                pt2 = (int(self.keypoint_2[match.trainIdx][0] + img1.shape[1]), int(self.keypoint_2[match.trainIdx][1]))

                cv2.line(combined_image, pt1, pt2, (0, 255, 0), 2)  # Зеленая линия
                cv2.circle(combined_image, pt1, 5, (0, 0, 255), -1)  # Красная точка на первом изображении
                cv2.circle(combined_image, pt2, 5, (0, 0, 255), -1)  # Красная точка на втором изображении

        cv2.imshow('Matches', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    img1 = "dataset/test/flowers-473.jpg"
    img2 = "dataset/test/00000.jpg"

    a = ImageMatching(img1, img2)

    a.compare_image()

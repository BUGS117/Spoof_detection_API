import os
import cv2
import numpy as np
import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


class SpoofDetector:
    def __init__(self, model_dir: str, device_id: int = 0):
        self.model_dir = model_dir
        self.device_id = device_id
        self.model_test = AntiSpoofPredict(device_id)
        self.image_cropper = CropImage()

    def check_image(self, image: np.ndarray) -> bool:
        """
        Check the aspect ratio of the image. Resizes if needed.
        """
        if image is None:
            print("Error: Image is None or could not be loaded!")
            return False
        height, width, _ = image.shape
        aspect_ratio = width / height
        if abs(aspect_ratio - 3 / 4) > 0.1:  # Relaxed tolerance for aspect ratio
            print("Image has an invalid aspect ratio. Resizing to 4:3.")
            new_width = int(height * 4 / 3)
            image = cv2.resize(image, (new_width, height))
        return True

    def predict(self, image: np.ndarray) -> dict:
        """
        Predict whether the image is a real or fake face.
        """
        if not self.check_image(image):
            return {"status": False, "message": "Invalid image input"}

        # Detect the face bounding box
        image_bbox = self.model_test.get_bbox(image)
        if image_bbox is None or len(image_bbox) != 4:
            return {"status": False, "message": "Failed to detect face in the image"}

        prediction = np.zeros((1, 3))
        test_speed = 0

        # Iterate through all models in the directory
        try:
            model_list = os.listdir(self.model_dir)
        except FileNotFoundError:
            return {"status": False, "message": f"Model directory not found: {self.model_dir}"}

        if not model_list:
            return {"status": False, "message": "No models found in the directory"}

        for model_name in model_list:
            try:
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False

                cropped_image = self.image_cropper.crop(**param)
                if cropped_image is None:
                    print(f"Warning: Cropping failed for model {model_name}")
                    continue

                start_time = time.time()
                prediction += self.model_test.predict(cropped_image, os.path.join(self.model_dir, model_name))
                test_speed += time.time() - start_time
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                continue

        # Compute the final prediction
        label = np.argmax(prediction)
        score = prediction[0][label] / len(model_list)
        result_label = "Real Face" if label == 1 else "Fake Face"
        return {
            "status": True,
            "label": result_label,
            "score": score,
            "processing_time": test_speed,
        }

    def visualize(self, image: np.ndarray, result: dict, image_bbox: list, output_path: str = None):
        """
        Draw bounding box and result text on the image for visualization.
        """
        if not result.get("status", False):
            print("Error: Unable to visualize as prediction failed.")
            return

        label_color = (255, 0, 0) if result["label"] == "Real Face" else (0, 0, 255)
        result_text = f"{result['label']} (Score: {result['score']:.2f})"

        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            label_color, 2
        )
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2
        )
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Result saved at {output_path}")
        else:
            cv2.imshow("Prediction", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define inputs
    image_path = "./images/sample/test_image.jpg"
    model_dir = "./resources/anti_spoof_models"
    output_path = "./images/sample/test_image_result.jpg"

    # Initialize detector
    detector = SpoofDetector(model_dir=model_dir, device_id=0)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Failed to load the image.")
    else:
        # Perform prediction
        result = detector.predict(image)
        if result["status"]:
            print(f"Prediction: {result['label']} (Score: {result['score']:.2f})")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            # Visualize and save the result
            detector.visualize(image, result, detector.model_test.get_bbox(image), output_path)
        else:
            print(f"Error: {result['message']}")

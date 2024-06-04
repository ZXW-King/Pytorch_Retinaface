import cv2

from onnx_export.detect_onnx import padding_resize


def main():
    image_path = '/media/xin/work/github_pro/face/Pytorch_Retinaface/curve/test.jpg'
    image = cv2.imread(image_path)

    # Ensure the image is in the correct format (HWC -> CHW)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = (640, 640)  # Example target size
    padding_color = (0, 0, 0)  # Example padding color (black)

    preprocessed_image = padding_resize(image, target_size, padding_color)
    print(preprocessed_image)

if __name__ == '__main__':
    main()
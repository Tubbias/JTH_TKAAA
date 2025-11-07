import argparse

import cv2
import gradio as gr
import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class YOLOPredictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_detection(self, image, conf, iou, image_size, use_tta):
        return self._predict_detections(conf, image, image_size, iou, use_tta)

    def _predict_detections(self, conf, image, image_size, iou, use_tta):
        results = self.model.predict(
            image,
            imgsz=image_size,
            conf=conf,
            iou=iou,
            augment=use_tta,
            agnostic_nms=True
        )

        res_detection_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

        return res_detection_image


def main():
    parser = argparse.ArgumentParser(description='Detect volleyball actions using YOLO model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to YOLO model')
    args = parser.parse_args()

    yolo_predictor = YOLOPredictor(args.model_path)

    gr_image_input = gr.Image(type='filepath', label="Input Image")
    gr_slider_confidence = gr.Slider(minimum=0.1, maximum=0.9, value=0.3, label="Confidence Threshold")
    gr_slider_iou = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, label="IOU Threshold")
    gr_slider_image_size = gr.Slider(minimum=320, maximum=1280, value=640, label="Image Size")
    gr_checkbox_use_tta = gr.Checkbox(label="Use Test Time Augmentation")
    gr_image_result = gr.Image(type='numpy', label="Output Image")

    iface = gr.Interface(
        fn=yolo_predictor.predict_detection,
        inputs=[
            gr_image_input,
            gr_slider_confidence,
            gr_slider_iou,
            gr_slider_image_size,
            gr_checkbox_use_tta
        ],
        outputs=gr_image_result,
        title="Volleyball Action Detection with YOLO",
        description="Volleyball Action Detection with YOLO"
    )

    iface.launch()


if __name__ == "__main__":
    main()

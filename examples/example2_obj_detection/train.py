import argparse
from ultralytics import YOLO


YOLOV8_N_MODEL = 'yolov8n.pt'
YOLOV8_S_MODEL = 'yolov8s.pt'

YOLO_SUPPORTED_MODELS = [
    YOLOV8_N_MODEL,
    YOLOV8_S_MODEL
]


def load_default_obj_det_hyperparameters():
    return {
        'hsv_h': 0.045,         # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.9,           # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.7,           # image HSV-Value augmentation (fraction)
        'degrees': 0.0,         # image rotation (+/- deg)
        'translate': 0.0,       # image translation (+/- fraction)
        'scale': 0.5,           # image scale (+/- gain)
        'shear': 0.0,           # image shear (+/- deg)
        'perspective': 0.0,     # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,          # image flip up-down (probability)
        'fliplr': 0.5,          # image flip left-right (probability)
        'mosaic': 0.5,          # image mosaic (probability)
        'mixup': 0.5,           # image mixup (probability)
        'copy_paste': 0.5,      # segment copy-paste (probability)
        'erase': 0.4,
        'lr0': 0.01,            # initial learning rate
    }


def main():
    parser = argparse.ArgumentParser(description='Yolo training script')
    parser.add_argument('--data-path', type=str, required=True, help='Path to yaml data file')
    parser.add_argument('--output-folder', type=str, required=False, default='runs/train', help='Path to output folder')
    parser.add_argument('--experiment-name', type=str, required=False, default='exp', help='Experiment name')
    parser.add_argument('--model-path', type=str, required=False, default=YOLOV8_N_MODEL,
                        help='Path to model file. Supported pre-trained models are YOLOv8 models {}'.format(
                            YOLO_SUPPORTED_MODELS))
    parser.add_argument('--image-size', type=int, required=False, default=640, help='Image size')
    parser.add_argument('--epochs', type=int, required=False, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, required=False, default=8, help='Batch size')
    args = parser.parse_args()

    model = YOLO(args.model_path)

    hyperparameter_dict = load_default_obj_det_hyperparameters()

    model.train(
        data=args.data_path,
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch_size,
        project=args.output_folder,
        name=args.experiment_name,
        hsv_h=hyperparameter_dict['hsv_h'],
        hsv_s=hyperparameter_dict['hsv_s'],
        hsv_v=hyperparameter_dict['hsv_v'],
        degrees=hyperparameter_dict['degrees'],
        translate=hyperparameter_dict['translate'],
        scale=hyperparameter_dict['scale'],
        shear=hyperparameter_dict['shear'],
        perspective=hyperparameter_dict['perspective'],
        flipud=hyperparameter_dict['flipud'],
        fliplr=hyperparameter_dict['fliplr'],
        mosaic=hyperparameter_dict['mosaic'],
        mixup=hyperparameter_dict['mixup'],
        copy_paste=hyperparameter_dict['copy_paste'],
        erasing=hyperparameter_dict['erase']
    )


if __name__ == "__main__":
    main()

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 on hail/wind damage dataset"
    )
    parser.add_argument("--model", default="yolo11s.pt",
                        help="Model weights to start from (default: yolo11s.pt)")
    parser.add_argument("--data", default="data.yaml",
                        help="Path to dataset YAML config (default: data.yaml)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--name", default="hail_yolo_v1",
                        help="Run name for saving results (default: hail_yolo_v1)")
    parser.add_argument("--project", default="runs",
                        help="Project directory for results (default: runs)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

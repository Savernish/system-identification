# main.py
import argparse
from src.training.train_model import train_fusion_model
from src.training.train_image_only import train_image_only_model
from src.training.train_signal_only import train_signal_only_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistem Tanımlama - Model Eğitimi")
    parser.add_argument(
        "--model",
        type=str,
        default="fusion",
        choices=["fusion", "image_only", "signal_only"],
        help="Eğitilecek model tipi (default: fusion)"
    )
    args = parser.parse_args()

    print("Sistem Başlatılıyor...")
    print(f"Model: {args.model}")

    if args.model == "fusion":
        train_fusion_model()
    elif args.model == "image_only":
        train_image_only_model()
    elif args.model == "signal_only":
        train_signal_only_model()
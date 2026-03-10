# main.py
import argparse
from src.training.train_model import train_fusion_model
from src.training.train_model_lstm import train_fusion_model_lstm
from src.training.train_image_only import train_image_only_model
from src.training.train_signal_only import train_signal_only_model
from src.training.train_signal_only_lstm import train_signal_only_lstm_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistem Tanımlama - Model Eğitimi")
    parser.add_argument(
        "--model",
        type=str,
        default="fusion",
        choices=["fusion", "fusion_lstm", "image_only", "signal_only", "signal_only_lstm"],
        help="Eğitilecek model tipi (default: fusion)"
    )
    args = parser.parse_args()

    print("Sistem Başlatılıyor...")
    print(f"Model: {args.model}")

    if args.model == "fusion":
        train_fusion_model()
    elif args.model == "fusion_lstm":
        train_fusion_model_lstm()
    elif args.model == "image_only":
        train_image_only_model()
    elif args.model == "signal_only":
        train_signal_only_model()
    elif args.model == "signal_only_lstm":
        train_signal_only_lstm_model()
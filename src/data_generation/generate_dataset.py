# src/data_generation/generate_dataset.py
"""
Python dataset generator for System Identification (Piroddi & Leva, 2007).
Generates step response data for 6 transfer function classes:
  Class1_FOPTD       : μ / (1 + sT)
  Class2_SOPTD       : μ / ((1+sT₁)(1+sT₂))
  Class3_NMP         : μ(1-sT_z) / ((1+sT₁)(1+sT₂))
  Class4_Underdamped : μ(1+sT_z) / ((1+sT₁)(1+sT₂)),  T_z > T₁, T₂
  Class5_HighOrder   : μωₙ² / (s²+2ζωₙs+ωₙ²)
  Class6_Integrator  : μωₙ² / ((sT+1)(s²+2ζωₙs+ωₙ²))

Each sample consists of:
  - Signal:  .npy file of shape [2, 2000]  (row 0 = u, row 1 = y)
  - Image:   .png file of size 448×228, binary (1-bit), white curve on black
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal as sp_signal
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# Simulation settings
# ──────────────────────────────────────────────────────────────
N_POINTS    = 2000          # Number of simulation steps
T_FINAL     = 100.0         # Total simulation time (seconds)
DT          = T_FINAL / N_POINTS   # 0.05 s per step
IMG_WIDTH   = 448
IMG_HEIGHT  = 228
NOISE_STD   = 0.02          # Noise as fraction of signal range

# ──────────────────────────────────────────────────────────────
# Transfer function generators
# ──────────────────────────────────────────────────────────────

def class1_foptd(rng):
    """Class1: G(s) = μ / (1 + sT)"""
    mu = rng.uniform(0.5, 5.0)
    T  = rng.uniform(1.0, 20.0)
    num = [mu]
    den = [T, 1.0]
    return sp_signal.TransferFunction(num, den)

def class2_soptd(rng):
    """Class2: G(s) = μ / ((1+sT₁)(1+sT₂))"""
    mu = rng.uniform(0.5, 5.0)
    T1 = rng.uniform(1.0, 15.0)
    T2 = rng.uniform(1.0, 15.0)
    num = [mu]
    den = np.polymul([T1, 1.0], [T2, 1.0])
    return sp_signal.TransferFunction(num, den)

def class3_nmp(rng):
    """Class3: G(s) = μ(1 - sT_z) / ((1+sT₁)(1+sT₂))"""
    mu = rng.uniform(0.5, 5.0)
    Tz = rng.uniform(0.5, 8.0)
    T1 = rng.uniform(1.0, 15.0)
    T2 = rng.uniform(1.0, 15.0)
    # Numerator: μ * (1 - s*Tz) = μ * [-Tz, 1]
    num = np.array([-Tz, 1.0]) * mu
    den = np.polymul([T1, 1.0], [T2, 1.0])
    return sp_signal.TransferFunction(num, den)

def class4_underdamped(rng):
    """Class4: G(s) = μ(1 + sT_z) / ((1+sT₁)(1+sT₂)),  T_z > T₁, T₂"""
    mu = rng.uniform(0.5, 5.0)
    T1 = rng.uniform(1.0, 10.0)
    T2 = rng.uniform(1.0, 10.0)
    # T_z must be greater than both T1 and T2
    Tz = max(T1, T2) + rng.uniform(1.0, 10.0)
    # Numerator: μ * (1 + s*Tz) = μ * [Tz, 1]
    num = np.array([Tz, 1.0]) * mu
    den = np.polymul([T1, 1.0], [T2, 1.0])
    return sp_signal.TransferFunction(num, den)

def class5_high_order(rng):
    """Class5: G(s) = μ·ωₙ² / (s² + 2ζωₙs + ωₙ²)"""
    mu  = rng.uniform(0.5, 5.0)
    wn  = rng.uniform(0.3, 3.0)
    zeta = rng.uniform(0.05, 0.7)   # underdamped
    num = [mu * wn**2]
    den = [1.0, 2*zeta*wn, wn**2]
    return sp_signal.TransferFunction(num, den)

def class6_integrator(rng):
    """Class6: G(s) = μ·ωₙ² / ((sT+1)(s² + 2ζωₙs + ωₙ²))"""
    mu   = rng.uniform(0.5, 5.0)
    T    = rng.uniform(1.0, 15.0)
    wn   = rng.uniform(0.3, 3.0)
    zeta = rng.uniform(0.05, 0.7)
    num = [mu * wn**2]
    second_order = [1.0, 2*zeta*wn, wn**2]
    den = np.polymul([T, 1.0], second_order)
    return sp_signal.TransferFunction(num, den)


# Class name → generator function
CLASS_GENERATORS = {
    'Class1_FOPTD':       class1_foptd,
    'Class2_SOPTD':       class2_soptd,
    'Class3_NMP':         class3_nmp,
    'Class4_Underdamped': class4_underdamped,
    'Class5_HighOrder':   class5_high_order,
    'Class6_Integrator':  class6_integrator,
}


# ──────────────────────────────────────────────────────────────
# Simulation & image generation
# ──────────────────────────────────────────────────────────────

def simulate_step_response(tf_system):
    """Simulate the step response and return (t, u, y) arrays of length N_POINTS."""
    t = np.linspace(0, T_FINAL, N_POINTS, endpoint=False)
    u = np.ones_like(t)
    _, y, _ = sp_signal.lsim(tf_system, U=u, T=t)
    return t, u, y


def add_noise(y, rng):
    """Add Gaussian noise proportional to the signal range."""
    y_range = np.ptp(y) if np.ptp(y) > 0 else 1.0
    noise = rng.normal(0, NOISE_STD * y_range, size=y.shape)
    return y + noise


def signal_to_binary_image(y):
    """
    Render the signal y as a binary image (white curve on black background).
    Matches the MATLAB-generated images: 448×228, mode '1' (1-bit), no axes.
    """
    fig = plt.figure(figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])  # fill entire figure
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    ax.plot(y, color='white', linewidth=0.8)
    ax.axis('off')
    ax.set_xlim(0, len(y) - 1)

    # Auto-scale y with small padding
    y_min, y_max = y.min(), y.max()
    y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.5
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Render to array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    plt.close(fig)

    # Convert RGBA → grayscale → binary
    gray = np.mean(img_array[:, :, :3], axis=2)
    binary = (gray > 128).astype(np.uint8) * 255

    # Resize to exact target dimensions & convert to 1-bit
    pil_img = Image.fromarray(binary, mode='L')
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    pil_img = pil_img.convert('1')  # 1-bit binary
    return pil_img


# ──────────────────────────────────────────────────────────────
# Dataset generation
# ──────────────────────────────────────────────────────────────

def generate_dataset(
    output_dir,
    samples_per_class_train=700,
    samples_per_class_val=150,
    samples_per_class_test=150,
    seed=42,
):
    """Generate the full dataset with train/val/test splits."""
    rng = np.random.default_rng(seed)

    split_counts = {
        'train': samples_per_class_train,
        'val':   samples_per_class_val,
        'test':  samples_per_class_test,
    }

    total_samples = sum(split_counts.values()) * len(CLASS_GENERATORS)
    print(f"{'='*60}")
    print(f"  VERİ SETİ OLUŞTURUCU")
    print(f"  Toplam örnek: {total_samples}")
    print(f"  Sınıflar: {list(CLASS_GENERATORS.keys())}")
    print(f"  Split: train={samples_per_class_train}, val={samples_per_class_val}, test={samples_per_class_test}")
    print(f"  Çıktı: {output_dir}")
    print(f"{'='*60}\n")

    generated = 0
    failed = 0

    for split_name, n_samples in split_counts.items():
        print(f"\n[{split_name.upper()}] — Sınıf başına {n_samples} örnek oluşturuluyor")

        for cls_name, gen_func in CLASS_GENERATORS.items():
            img_dir = os.path.join(output_dir, 'images', split_name, cls_name)
            sig_dir = os.path.join(output_dir, 'signals', split_name, cls_name)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(sig_dir, exist_ok=True)

            pbar = tqdm(range(1, n_samples + 1), desc=f"  {cls_name}", leave=False)
            for i in pbar:
                sample_name = f"sample_{i:04d}"

                try:
                    # 1. Generate random transfer function
                    tf_sys = gen_func(rng)

                    # 2. Simulate step response
                    t, u, y = simulate_step_response(tf_sys)

                    # 3. Check for unstable/diverging systems and retry
                    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e6:
                        # Retry with new parameters
                        for _ in range(10):
                            tf_sys = gen_func(rng)
                            t, u, y = simulate_step_response(tf_sys)
                            if np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e6:
                                break
                        else:
                            failed += 1
                            continue

                    # 4. Add noise
                    y_noisy = add_noise(y, rng)

                    # 5. Save signal as .npy [2, 2000]
                    sig_array = np.vstack((u, y_noisy)).astype(np.float32)
                    np.save(os.path.join(sig_dir, f"{sample_name}.npy"), sig_array)

                    # 6. Generate and save binary image
                    img = signal_to_binary_image(y_noisy)
                    img.save(os.path.join(img_dir, f"{sample_name}.png"))

                    generated += 1

                except Exception as e:
                    failed += 1
                    tqdm.write(f"    Hata ({cls_name}/{sample_name}): {e}")

    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI")
    print(f"  Başarılı: {generated}")
    print(f"  Başarısız: {failed}")
    print(f"  Çıktı dizini: {output_dir}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    parser = argparse.ArgumentParser(description="Sistem Tanımlama Veri Seti Oluşturucu")
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(project_root, 'data'),
        help="Çıktı dizini (default: data/)"
    )
    parser.add_argument('--train', type=int, default=700, help="Sınıf başına train örnek sayısı")
    parser.add_argument('--val',   type=int, default=150, help="Sınıf başına val örnek sayısı")
    parser.add_argument('--test',  type=int, default=150, help="Sınıf başına test örnek sayısı")
    parser.add_argument('--seed',  type=int, default=42,  help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        samples_per_class_train=args.train,
        samples_per_class_val=args.val,
        samples_per_class_test=args.test,
        seed=args.seed,
    )

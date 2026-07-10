# src/data_generation/generate_dataset.py
"""
Dataset generator for System Identification (Piroddi & Leva, 2007).
Generates step response data for 6 transfer function classes:
  Class1_FirstOrder  : mu / (1 + sT)
  Class2_Overdamped  : mu / ((1+sT1)(1+sT2))
  Class3_NMP         : mu(1-sT_z) / ((1+sT1)(1+sT2))
  Class4_Underdamped : mu*wn^2 / (s^2 + 2*zeta*wn*s + wn^2),  zeta < 1
  Class5_HighOrder   : mu / ((1+sT1)(1+sT2)(1+sT3)(1+sT4))  — 4th order
  Class6_Integrator  : mu / (s*(1+sT))  — true integrator, ramp response

Variant system
--------------
Each "special type" (e.g. dead time) has a FIXED fraction of samples it applies
to within every class and every split.  Only the magnitude of the variant is
randomized, not whether the sample has it.  Fractions are set in VARIANT_CONFIG
before generation begins.

Outputs per sample
------------------
  Signal : .npy  shape [2, 2000]  (row 0 = u, row 1 = y)
  Image  : .png  448x228, binary 1-bit, white curve on black

Outputs per split
-----------------
  metadata_{split}.csv  one row per sample:
      filename, class, split, has_dead_time, dead_time_value, noise_std
"""

import os
import csv
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
N_POINTS   = 2000
T_FINAL    = 100.0
DT         = T_FINAL / N_POINTS   # 0.05 s
IMG_WIDTH  = 448
IMG_HEIGHT = 228

# ──────────────────────────────────────────────────────────────
# Variant configuration
# Add new special types here. Each entry must have:
#   fraction  – exact fraction of samples in EVERY class+split that get
#               this variant (0.0 = none, 1.0 = all)
#   min_value – lower bound of the randomised magnitude
#   max_value – upper bound of the randomised magnitude
# ──────────────────────────────────────────────────────────────
VARIANT_CONFIG = {
    'dead_time': {
        'fraction':  0.25,   # 25 % with dead time, 75 % without
        'min_value': 1.0,    # seconds  (randomised within this range)
        'max_value': 15.0,
    },
    'noise_level': {
        'high_fraction': 0.25,  # 25 % get high noise, 75 % get low noise
        'low_std':  0.02,       # noise std for the majority (2 %)
        'high_std': 0.05,       # noise std for the minority (5 %)
    },
}

# ──────────────────────────────────────────────────────────────
# Transfer function generators
# ──────────────────────────────────────────────────────────────

def class1_first_order(rng):
    """G(s) = mu / (1 + sT)"""
    mu = rng.uniform(0.5, 5.0)
    T  = rng.uniform(1.0, 20.0)
    return sp_signal.TransferFunction([mu], [T, 1.0])

def class2_overdamped(rng):
    """G(s) = mu / ((1+sT1)(1+sT2))"""
    mu = rng.uniform(0.5, 5.0)
    T1 = rng.uniform(1.0, 15.0)
    T2 = rng.uniform(1.0, 15.0)
    return sp_signal.TransferFunction([mu], np.polymul([T1, 1.0], [T2, 1.0]))

def class3_nmp(rng):
    """G(s) = mu(1 - sT_z) / ((1+sT1)(1+sT2))"""
    mu = rng.uniform(0.5, 5.0)
    Tz = rng.uniform(0.5, 8.0)
    T1 = rng.uniform(1.0, 15.0)
    T2 = rng.uniform(1.0, 15.0)
    num = np.array([-Tz, 1.0]) * mu
    return sp_signal.TransferFunction(num, np.polymul([T1, 1.0], [T2, 1.0]))

def class4_underdamped(rng):
    """G(s) = mu*wn^2 / (s^2 + 2*zeta*wn*s + wn^2),  zeta < 1"""
    mu   = rng.uniform(0.5, 5.0)
    wn   = rng.uniform(0.3, 3.0)
    zeta = rng.uniform(0.05, 0.7)
    return sp_signal.TransferFunction([mu * wn**2], [1.0, 2*zeta*wn, wn**2])

def class5_high_order(rng):
    """G(s) = mu / ((1+sT1)(1+sT2)(1+sT3)(1+sT4))  — 4th order"""
    mu = rng.uniform(0.5, 5.0)
    T1 = rng.uniform(1.0, 10.0)
    T2 = rng.uniform(1.0, 10.0)
    T3 = rng.uniform(1.0, 10.0)
    T4 = rng.uniform(1.0, 10.0)
    den = np.polymul(np.polymul([T1, 1.0], [T2, 1.0]), np.polymul([T3, 1.0], [T4, 1.0]))
    return sp_signal.TransferFunction([mu], den)

def class6_integrator(rng):
    """G(s) = mu / (s*(1+sT))  — true integrator, ramp step response"""
    mu = rng.uniform(0.02, 0.15)   # small to keep output bounded in T_FINAL=100s
    T  = rng.uniform(1.0, 15.0)
    # s*(1+sT) = T*s^2 + s + 0
    return sp_signal.TransferFunction([mu], [T, 1.0, 0.0])

CLASS_GENERATORS = {
    'Class1_FirstOrder':  class1_first_order,
    'Class2_Overdamped':  class2_overdamped,
    'Class3_NMP':         class3_nmp,
    'Class4_Underdamped': class4_underdamped,
    'Class5_HighOrder':   class5_high_order,
    'Class6_Integrator':  class6_integrator,
}

# ──────────────────────────────────────────────────────────────
# Simulation helpers
# ──────────────────────────────────────────────────────────────

def simulate_step_response(tf_system):
    t = np.linspace(0, T_FINAL, N_POINTS, endpoint=False)
    u = np.ones_like(t)
    _, y, _ = sp_signal.lsim(tf_system, U=u, T=t)
    return t, u, y

def is_valid(y):
    return np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e6

def add_noise(y, rng, noise_std):
    y_range = np.ptp(y) if np.ptp(y) > 0 else 1.0
    return y + rng.normal(0, noise_std * y_range, size=y.shape)

def apply_dead_time(y, dead_time_secs):
    """Shift y right by dead_time_secs, pad leading values with zero."""
    delay_steps = int(round(dead_time_secs / DT))
    y_delayed = np.zeros_like(y)
    if delay_steps < len(y):
        y_delayed[delay_steps:] = y[: len(y) - delay_steps]
    return y_delayed

def compute_system_properties(y):
    """
    Compute step response metrics from a clean signal (after dead time, before noise).
    Returns: steady_state_gain, rise_time, settling_time, overshoot_pct
    """
    n = len(y)
    t = np.arange(n) * DT

    y_ss = float(np.mean(y[int(0.9 * n):]))
    if abs(y_ss) < 1e-9:
        return {'steady_state_gain': 0.0, 'rise_time': float(T_FINAL),
                'settling_time': float(T_FINAL), 'overshoot_pct': 0.0}

    # Rise time: first 10% → first 90% crossing toward final value
    y10, y90 = 0.1 * y_ss, 0.9 * y_ss
    if y_ss > 0:
        idx10 = int(np.argmax(y >= y10))
        idx90 = int(np.argmax(y >= y90))
    else:
        idx10 = int(np.argmax(y <= y10))
        idx90 = int(np.argmax(y <= y90))
    rise_time = float(t[idx90] - t[idx10]) if idx90 > idx10 else float(T_FINAL)

    # Settling time: last sample outside 2% band around y_ss
    outside = np.where(np.abs(y - y_ss) > 0.02 * abs(y_ss))[0]
    settling_time = float(t[outside[-1]]) if len(outside) > 0 else 0.0

    # Overshoot %: peak exceedance beyond y_ss
    if y_ss > 0:
        overshoot_pct = float(max(0.0, (np.max(y) - y_ss) / y_ss * 100))
    else:
        overshoot_pct = float(max(0.0, (y_ss - np.min(y)) / abs(y_ss) * 100))

    return {
        'steady_state_gain': round(y_ss, 4),
        'rise_time':         round(rise_time, 4),
        'settling_time':     round(settling_time, 4),
        'overshoot_pct':     round(overshoot_pct, 4),
    }

def signal_to_binary_image(y):
    fig = plt.figure(figsize=(IMG_WIDTH / 100, IMG_HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.plot(y, color='white', linewidth=0.8)
    ax.axis('off')
    ax.set_xlim(0, len(y) - 1)
    y_min, y_max = y.min(), y.max()
    margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.5
    ax.set_ylim(y_min - margin, y_max + margin)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    plt.close(fig)
    gray   = np.mean(img_array[:, :, :3], axis=2)
    binary = (gray > 128).astype(np.uint8) * 255
    pil_img = Image.fromarray(binary, mode='L')
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    return pil_img.convert('1')

# ──────────────────────────────────────────────────────────────
# Variant assignment
# Returns a list of dicts, one per sample index (0-based).
# Fractions are exact: int(round(n * fraction)) samples get each variant.
# Assignment is shuffled deterministically so DT samples are not clustered.
# ──────────────────────────────────────────────────────────────

def assign_variants(n_samples, rng, variant_config):
    """
    Returns list of length n_samples.
    Each element: {'dead_time': float, 'noise_std': float}
      dead_time  – 0.0 means no dead time; >0 is the delay in seconds
      noise_std  – fraction of signal range used as Gaussian noise std
    """
    noise_cfg = variant_config.get('noise_level', {})
    default_noise = noise_cfg.get('low_std', 0.02)
    assignments = [{'dead_time': 0.0, 'noise_std': default_noise}
                   for _ in range(n_samples)]

    # ── dead time ──────────────────────────────────────────────
    dt_cfg = variant_config.get('dead_time')
    if dt_cfg and dt_cfg.get('fraction', 0.0) > 0.0:
        n_with_dt = int(round(n_samples * dt_cfg['fraction']))
        dt_indices = set(rng.permutation(n_samples)[:n_with_dt])
        for i in dt_indices:
            assignments[i]['dead_time'] = float(
                rng.uniform(dt_cfg['min_value'], dt_cfg['max_value'])
            )

    # ── noise level ────────────────────────────────────────────
    if noise_cfg and noise_cfg.get('high_fraction', 0.0) > 0.0:
        n_high = int(round(n_samples * noise_cfg['high_fraction']))
        high_indices = set(rng.permutation(n_samples)[:n_high])
        for i in high_indices:
            assignments[i]['noise_std'] = noise_cfg['high_std']

    return assignments

# ──────────────────────────────────────────────────────────────
# Dataset generation
# ──────────────────────────────────────────────────────────────

def generate_dataset(
    output_dir,
    samples_per_class_train=700,
    samples_per_class_val=150,
    samples_per_class_test=150,
    variant_config=None,
    seed=42,
):
    if variant_config is None:
        variant_config = VARIANT_CONFIG

    rng = np.random.default_rng(seed)

    split_counts = {
        'train': samples_per_class_train,
        'val':   samples_per_class_val,
        'test':  samples_per_class_test,
    }

    dt_cfg    = variant_config.get('dead_time', {})
    noise_cfg = variant_config.get('noise_level', {})
    dt_fraction    = dt_cfg.get('fraction', 0.0)
    noise_high_frac = noise_cfg.get('high_fraction', 0.0)

    total_samples = sum(split_counts.values()) * len(CLASS_GENERATORS)
    n_classes     = len(CLASS_GENERATORS)

    print(f"\n{'='*65}")
    print(f"  DATASET GENERATOR")
    print(f"  Total samples      : {total_samples}")
    print(f"  Classes            : {n_classes}")
    print(f"  Split              : train={samples_per_class_train}  val={samples_per_class_val}  test={samples_per_class_test}")
    print(f"  Dead time          : {dt_fraction*100:.0f}% with  (range {dt_cfg.get('min_value',0):.1f}–{dt_cfg.get('max_value',0):.1f} s)")
    print(f"  Noise level        : {noise_high_frac*100:.0f}% high ({noise_cfg.get('high_std',0.05)*100:.0f}%)  /  {(1-noise_high_frac)*100:.0f}% low ({noise_cfg.get('low_std',0.02)*100:.0f}%)")
    print(f"  Output             : {output_dir}")
    print(f"{'='*65}\n")

    generated = 0
    failed    = 0

    for split_name, n_samples in split_counts.items():
        n_dt     = int(round(n_samples * dt_fraction))
        n_high_n = int(round(n_samples * noise_high_frac))
        print(f"[{split_name.upper()}]  {n_samples} samples/class  "
              f"| dead time: {n_dt} with / {n_samples-n_dt} without  "
              f"| noise: {n_high_n} high / {n_samples-n_high_n} low")

        split_metadata = []   # rows for metadata CSV

        for cls_name, gen_func in CLASS_GENERATORS.items():
            img_dir = os.path.join(output_dir, 'images',  split_name, cls_name)
            sig_dir = os.path.join(output_dir, 'signals', split_name, cls_name)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(sig_dir, exist_ok=True)

            # Decide which samples get dead time BEFORE the generation loop
            variant_assignments = assign_variants(n_samples, rng, variant_config)

            pbar = tqdm(range(n_samples), desc=f"  {cls_name}", leave=False)
            for i in pbar:
                sample_name = f"sample_{i+1:04d}"
                assignment  = variant_assignments[i]
                dead_time   = assignment['dead_time']
                noise_std   = assignment['noise_std']

                try:
                    # Generate and validate transfer function
                    tf_sys = gen_func(rng)
                    t, u, y = simulate_step_response(tf_sys)

                    if not is_valid(y):
                        for _ in range(10):
                            tf_sys = gen_func(rng)
                            t, u, y = simulate_step_response(tf_sys)
                            if is_valid(y):
                                break
                        else:
                            failed += 1
                            continue

                    # Apply dead time (if assigned)
                    if dead_time > 0.0:
                        y = apply_dead_time(y, dead_time)

                    # Compute properties from clean signal (after DT, before noise)
                    props = compute_system_properties(y)

                    # Add measurement noise
                    y_noisy = add_noise(y, rng, noise_std)

                    # Save signal [2, N_POINTS]
                    sig_array = np.vstack((u, y_noisy)).astype(np.float32)
                    np.save(os.path.join(sig_dir, f"{sample_name}.npy"), sig_array)

                    # Save binary image
                    img = signal_to_binary_image(y_noisy)
                    img.save(os.path.join(img_dir, f"{sample_name}.png"))

                    # Record metadata
                    split_metadata.append({
                        'filename':        sample_name,
                        'class':           cls_name,
                        'split':           split_name,
                        'has_dead_time':   dead_time > 0.0,
                        'dead_time_value': round(dead_time, 4),
                        'noise_std':       noise_std,
                        **props,
                    })

                    generated += 1

                except Exception as e:
                    failed += 1
                    tqdm.write(f"    Error ({cls_name}/{sample_name}): {e}")

        # Write metadata CSV for this split
        meta_path = os.path.join(output_dir, f"metadata_{split_name}.csv")
        fieldnames = ['filename', 'class', 'split', 'has_dead_time', 'dead_time_value',
                      'noise_std', 'steady_state_gain', 'rise_time', 'settling_time', 'overshoot_pct']
        with open(meta_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_metadata)
        print(f"  -> Metadata saved: {meta_path}  ({len(split_metadata)} rows)\n")

    print(f"{'='*65}")
    print(f"  DONE")
    print(f"  Generated : {generated}")
    print(f"  Failed    : {failed}")
    print(f"  Output    : {output_dir}")
    print(f"{'='*65}\n")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    parser = argparse.ArgumentParser(description="System Identification Dataset Generator")
    parser.add_argument('--output',             type=str,   default=os.path.join(project_root, 'data'))
    parser.add_argument('--train',              type=int,   default=700,  help="Samples per class (train)")
    parser.add_argument('--val',                type=int,   default=150,  help="Samples per class (val)")
    parser.add_argument('--test',               type=int,   default=150,  help="Samples per class (test)")
    parser.add_argument('--dead-time-fraction',  type=float, default=0.25, help="Fraction of samples with dead time (0.0–1.0)")
    parser.add_argument('--dead-time-min',       type=float, default=1.0,  help="Min dead time in seconds")
    parser.add_argument('--dead-time-max',       type=float, default=15.0, help="Max dead time in seconds")
    parser.add_argument('--noise-high-fraction', type=float, default=0.25, help="Fraction of samples with high noise (0.0–1.0)")
    parser.add_argument('--noise-low-std',       type=float, default=0.02, help="Noise std for low-noise samples (default 2%%)")
    parser.add_argument('--noise-high-std',      type=float, default=0.05, help="Noise std for high-noise samples (default 5%%)")
    parser.add_argument('--seed',                type=int,   default=42)
    args = parser.parse_args()

    variant_config = {
        'dead_time': {
            'fraction':  args.dead_time_fraction,
            'min_value': args.dead_time_min,
            'max_value': args.dead_time_max,
        },
        'noise_level': {
            'high_fraction': args.noise_high_fraction,
            'low_std':       args.noise_low_std,
            'high_std':      args.noise_high_std,
        },
    }

    generate_dataset(
        output_dir=args.output,
        samples_per_class_train=args.train,
        samples_per_class_val=args.val,
        samples_per_class_test=args.test,
        variant_config=variant_config,
        seed=args.seed,
    )

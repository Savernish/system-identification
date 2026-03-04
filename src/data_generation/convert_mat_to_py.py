# src/data_generation/convert_mat_to_npy.py
import os
import glob
import scipy.io as sio
import numpy as np
from tqdm import tqdm

def convert_dataset():
    # Proje kök dizinini bulur (Bu dosyanın 2 klasör üstü)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    sig_dir = os.path.join(project_root, 'data', 'signals')
    
    splits = ['train', 'val', 'test']
    # Senin orijinal datandaki klasör isimleri neyse birebir aynı olmalı
    classes = [
        'Underdamped', 'Overdamped', 'Class1_FOPTD', 
        'Class2_SOPTD', 'Class3_NMP', 'Class4_Underdamped', 
        'Class5_HighOrder', 'Class6_Integrator'
    ] 
    # Not: Eski 2'li sınıfların (Underdamped/Overdamped) veya 6'lı sınıfların
    # hangisi orijinal datanda varsa bu listeyi ona göre düzenleyebilirsin.
    # Ben hata vermesin diye klasörde ne bulursa çevirecek bir mantık yazıyorum:
    
    total_converted = 0
    print("MATLAB (.mat) dosyaları yüksek hızlı NumPy (.npy) formatına dönüştürülüyor...\n")
    
    for split in splits:
        split_dir = os.path.join(sig_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        # O split içindeki tüm sınıf klasörlerini bul
        class_folders = [f.name for f in os.scandir(split_dir) if f.is_dir()]
        
        for cls_name in class_folders:
            folder_path = os.path.join(split_dir, cls_name)
            mat_files = glob.glob(os.path.join(folder_path, '*.mat'))
            
            if not mat_files:
                continue
                
            print(f"[{split.upper()}] {cls_name} işleniyor...")
            
            for mat_path in tqdm(mat_files, leave=False):
                try:
                    mat_data = sio.loadmat(mat_path)
                    u_sig = mat_data['u'].flatten()
                    y_sig = mat_data['y'].flatten()
                    
                    # [2, 2000] boyutunda birleştir ve kaydet
                    sig_array = np.vstack((u_sig, y_sig)).astype(np.float32)
                    npy_path = mat_path.replace('.mat', '.npy')
                    np.save(npy_path, sig_array)
                    
                    total_converted += 1
                except Exception as e:
                    print(f"\nHata ({mat_path}): {e}")
                    
    print(f"\nİşlem Tamam! Toplam {total_converted} adet dosya .npy formatına çevrildi.")

if __name__ == "__main__":
    convert_dataset()
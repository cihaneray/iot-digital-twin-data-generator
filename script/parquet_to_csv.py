import pandas as pd
import sys
import os

def parquet_to_csv(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Hata: {input_path} bulunamadı.")
        return
    
    # Çıktı dosyası belirtilmemişse otomatik isim oluştur
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".csv"

    try:
        print("Parquet dosyası okunuyor...")
        df = pd.read_parquet(input_path)

        print("CSV olarak kaydediliyor...")
        df.to_csv(output_path, index=False)

        print(f"Başarılı! Çıktı dosyası: {output_path}")

    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python script.py input.parquet [output.csv]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        parquet_to_csv(input_file, output_file)
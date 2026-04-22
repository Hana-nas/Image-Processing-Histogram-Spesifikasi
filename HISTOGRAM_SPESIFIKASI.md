# 📊 Histogram Spesifikasi (Histogram Matching)

> **Pengolahan Citra Digital** — Penjelasan Konsep & Implementasi Kode

---

## 📌 Apa itu Histogram Spesifikasi?

**Histogram Spesifikasi** (atau *Histogram Matching*) adalah teknik pengolahan citra yang bertujuan mengubah distribusi intensitas piksel sebuah gambar (*gambar input*) agar mendekati distribusi intensitas gambar lainnya (*gambar referensi*).

Berbeda dengan **Histogram Equalization** yang memratakan distribusi ke bentuk seragam (uniform), histogram spesifikasi memungkinkan kita menargetkan distribusi **sembarang** sesuai referensi yang kita pilih.

```
Gambar Input  ──►  [Histogram Matching]  ──►  Gambar Hasil
                          ▲
                   Gambar Referensi
                   (target distribusi)
```

---

## 🎯 Tujuan & Kegunaan

| Kegunaan | Contoh Penerapan |
|---|---|
| Penyesuaian pencahayaan | Menyamakan kecerahan foto yang diambil pada kondisi berbeda |
| Normalisasi citra medis | Menyeragamkan kontras gambar MRI/CT scan |
| Pengolahan citra satelit | Mencocokkan distribusi warna antar citra dari waktu berbeda |
| Preprocessing deep learning | Normalisasi dataset gambar sebelum training |

---

## 🧮 Konsep Matematika

### 1. Histogram

Histogram adalah representasi frekuensi setiap nilai intensitas piksel (0–255) pada sebuah gambar.

$$h(r_k) = n_k$$

- $r_k$ = nilai intensitas ke-$k$ (0 hingga 255)
- $n_k$ = jumlah piksel dengan intensitas $r_k$

### 2. CDF (Cumulative Distribution Function)

CDF histogram adalah jumlah kumulatif dari histogram, yang kemudian dinormalisasi ke rentang [0, 1].

$$CDF(r_k) = \frac{\sum_{j=0}^{k} h(r_j) - CDF_{min}}{N_{total} - CDF_{min}}$$

- $N_{total}$ = total jumlah piksel dalam gambar
- $CDF_{min}$ = nilai CDF minimum yang bukan nol

### 3. Pemetaan Intensitas (Look-Up Table)

Untuk setiap nilai intensitas $i$ pada gambar input, cari nilai intensitas $j$ pada gambar referensi yang memenuhi:

$$j = \arg\min_{j} \left| CDF_{ref}(j) - CDF_{src}(i) \right|$$

Artinya: temukan intensitas referensi yang CDF-nya **paling mendekati** CDF input.

---

## 🔄 Alur Algoritma Langkah demi Langkah

```
 INPUT IMAGE          REFERENCE IMAGE
      │                     │
      ▼                     ▼
 Hitung Hist.         Hitung Hist.
      │                     │
      ▼                     ▼
 Hitung CDF_src       Hitung CDF_ref
      │                     │
      └──────────┬──────────┘
                 ▼
         Buat Tabel LUT
     (matching CDF src → ref)
                 │
                 ▼
     Terapkan LUT pada Input
                 │
                 ▼
           OUTPUT IMAGE
```

---

## 💻 Implementasi Kode (Python)

### Langkah 1 — Hitung CDF

```python
import numpy as np

def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    Hitung CDF (Cumulative Distribution Function) dari histogram.

    Args:
        histogram: Array 1-D dengan 256 elemen (frekuensi tiap intensitas).

    Returns:
        cdf_normalized: CDF yang telah dinormalisasi ke rentang [0, 1].
    """
    # Jumlah kumulatif dari histogram
    cdf = histogram.cumsum()

    # Ambil nilai minimum CDF yang bukan nol (hindari pembagian dengan nol)
    cdf_min = cdf[cdf > 0].min()

    # Total piksel = nilai akhir CDF
    total_pixels = cdf[-1]

    # Normalisasi ke [0, 1]
    cdf_normalized = (cdf - cdf_min) / (total_pixels - cdf_min)

    # Pastikan tidak ada nilai negatif (clipping)
    cdf_normalized = np.clip(cdf_normalized, 0, 1)

    return cdf_normalized
```

---

### Langkah 2 — Histogram Matching Satu Channel

```python
def histogram_specification_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Lakukan histogram matching pada satu channel (grayscale atau satu channel RGB).

    Args:
        src: Array 2-D gambar sumber (satu channel), dtype uint8.
        ref: Array 2-D gambar referensi (satu channel), dtype uint8.

    Returns:
        Array 2-D hasil histogram matching, dtype uint8.
    """
    # Langkah 1: Hitung histogram masing-masing gambar (256 bin)
    hist_src, _ = np.histogram(src.flatten(), bins=256, range=(0, 256))
    hist_ref, _ = np.histogram(ref.flatten(), bins=256, range=(0, 256))

    # Langkah 2: Hitung CDF untuk kedua gambar
    cdf_src = compute_cdf(hist_src)
    cdf_ref = compute_cdf(hist_ref)

    # Langkah 3: Buat Look-Up Table (LUT)
    # Untuk setiap intensitas i (0-255) pada src,
    # cari intensitas j pada ref yang CDF-nya paling dekat dengan CDF src[i]
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(cdf_ref - cdf_src[i])   # selisih CDF
        lut[i] = np.argmin(diff)               # ambil indeks dengan selisih terkecil

    # Langkah 4: Terapkan LUT pada gambar sumber
    # Setiap nilai piksel diganti sesuai pemetaan LUT
    return lut[src]
```

---

### Langkah 3 — Histogram Matching Gambar Lengkap (Grayscale & RGB)

```python
import cv2

def histogram_specification(src_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    """
    Histogram matching lengkap — mendukung gambar grayscale dan RGB.

    Untuk gambar grayscale : proses satu channel langsung.
    Untuk gambar RGB/BGR   : proses tiap channel secara terpisah (B, G, R),
                             lalu gabungkan kembali.

    Args:
        src_img: Gambar sumber (NumPy array, BGR atau GRAY).
        ref_img: Gambar referensi (NumPy array, BGR atau GRAY).

    Returns:
        Gambar hasil histogram matching.
    """
    if len(src_img.shape) == 2:
        # ── Gambar Grayscale ─────────────────────────────────
        if len(ref_img.shape) == 3:
            # Konversi referensi ke grayscale jika perlu
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        return histogram_specification_channel(src_img, ref_img)

    else:
        # ── Gambar BGR (Color) ────────────────────────────────
        if len(ref_img.shape) == 2:
            # Konversi referensi ke BGR jika perlu
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

        # Proses setiap channel secara independen
        result_channels = []
        for c in range(3):  # 0=Blue, 1=Green, 2=Red
            matched = histogram_specification_channel(
                src_img[:, :, c],
                ref_img[:, :, c]
            )
            result_channels.append(matched)

        # Gabungkan kembali menjadi gambar BGR
        return cv2.merge(result_channels)
```

---

### Langkah 4 — Visualisasi Histogram dengan Matplotlib

```python
import matplotlib
matplotlib.use('Agg')   # Backend non-GUI untuk server
import matplotlib.pyplot as plt
import io, base64

def plot_histograms(src_img, ref_img, out_img):
    """
    Buat plot histogram untuk tiga gambar: sumber, referensi, dan hasil.
    Mengembalikan gambar plot dalam format Base64 string (PNG).
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    titles = ['Input', 'Referensi', 'Hasil']
    images = [src_img, ref_img, out_img]

    for ax, img, title in zip(axes, images, titles):
        ax.set_title(f'Histogram — Gambar {title}')

        if len(img.shape) == 2:
            # Grayscale: satu garis histogram
            hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
            ax.plot(hist, color='white')
        else:
            # Color: tiga garis per channel (B, G, R)
            colors = ['blue', 'green', 'red']
            labels = ['Blue', 'Green', 'Red']
            for c, (color, label) in enumerate(zip(colors, labels)):
                hist, _ = np.histogram(img[:, :, c].flatten(), bins=256, range=(0, 256))
                ax.plot(hist, color=color, label=label)
            ax.legend()

    plt.tight_layout()

    # Simpan ke buffer memory → encode ke Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
```

---

### Langkah 5 — Contoh Penggunaan Lengkap

```python
import cv2

# Baca gambar (OpenCV membaca dalam format BGR)
src_img = cv2.imread('gambar_input.jpg')
ref_img = cv2.imread('gambar_referensi.jpg')

# Lakukan histogram matching
result = histogram_specification(src_img, ref_img)

# Simpan hasil
cv2.imwrite('gambar_hasil.jpg', result)

print("Histogram matching selesai!")
print(f"Ukuran input  : {src_img.shape}")
print(f"Ukuran hasil  : {result.shape}")
```

---

## 📊 Contoh Visual Alur CDF Matching

Misalkan untuk satu channel grayscale:

```
Intensitas │  CDF_src  │  CDF_ref  │  Mapping LUT
───────────┼───────────┼───────────┼──────────────
     0     │   0.00    │   0.00    │   0  → 0
    50     │   0.15    │   0.14 ◄──┤   50 → 49
   100     │   0.40    │   0.39 ◄──┤   100 → 99
   150     │   0.70    │   0.71 ◄──┤   150 → 151
   200     │   0.90    │   0.89 ◄──┤   200 → 198
   255     │   1.00    │   1.00    │   255 → 255
```

> Setiap intensitas pada gambar input dipetakan ke intensitas referensi
> yang memiliki **nilai CDF paling dekat**.

---

## ⚙️ Perbedaan dengan Teknik Lain

| Teknik | Distribusi Target | Kelebihan | Kekurangan |
|---|---|---|---|
| **Histogram Equalization** | Distribusi seragam (uniform) | Sederhana, otomatis | Tidak bisa dikontrol |
| **Histogram Spesifikasi** | Distribusi gambar referensi | Target fleksibel | Butuh gambar referensi |
| **Contrast Stretching** | Linier min-max | Sangat cepat | Sensitif terhadap outlier |

---

## 🗂️ Struktur File Proyek

```
PC_HistogramSpesifikasi/
├── app.py                        ← Backend Flask (routing + logika)
├── requirements.txt              ← Daftar dependensi Python
├── HISTOGRAM_SPESIFIKASI.md      ← Dokumentasi ini
├── templates/
│   └── index.html                ← Frontend (HTML + CSS + JS)
├── static/                       ← Asset statis (opsional)
└── uploads/                      ← Folder penyimpanan upload
```

---

## 📦 Dependensi

```txt
flask>=2.3.0          # Web framework Python
opencv-python>=4.8.0  # Baca/tulis & manipulasi gambar
numpy>=1.24.0         # Komputasi array & histogram
matplotlib>=3.7.0     # Visualisasi plot histogram
werkzeug>=2.3.0       # Utilitas Flask (file upload)
```

Install semua sekaligus:
```bash
py -m pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan Aplikasi

```bash
# 1. Masuk ke folder proyek
cd C:\Users\ASUS\Documents\PC_HistogramSpesifikasi

# 2. Install dependensi (hanya perlu sekali)
py -m pip install -r requirements.txt

# 3. Jalankan server Flask
py app.py

# 4. Buka di browser
#    http://127.0.0.1:5000
```

---

*Dibuat dengan Flask · OpenCV · NumPy · Matplotlib*

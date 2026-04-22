"""
======================================================
  Aplikasi Histogram Spesifikasi (Histogram Matching)
  Backend: Flask + OpenCV + NumPy + Matplotlib
======================================================
"""

import os
import io
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI agar bisa dipakai di server
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# ──────────────────────────────────────────────────────────────
#  Konfigurasi Aplikasi Flask
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)

# Folder untuk menyimpan file yang diupload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ekstensi file gambar yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maks 16 MB per file


# ──────────────────────────────────────────────────────────────
#  Fungsi Utilitas
# ──────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Cek apakah ekstensi file termasuk yang diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_bgr: np.ndarray) -> str:
    """
    Konversi array NumPy gambar (BGR atau GRAY) ke string Base64 PNG
    agar bisa langsung di-embed di HTML tanpa menyimpan file tambahan.
    """
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        raise ValueError("Gagal mengonversi gambar ke PNG")
    return base64.b64encode(buffer).decode('utf-8')


# ──────────────────────────────────────────────────────────────
#  Fungsi Inti: Histogram Specification
# ──────────────────────────────────────────────────────────────

def compute_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    Hitung CDF (Cumulative Distribution Function) dari sebuah histogram.

    Args:
        histogram: Array 1-D histogram dengan 256 bin (nilai intensitas 0-255).

    Returns:
        cdf_normalized: CDF yang sudah dinormalisasi ke rentang [0, 1].
    """
    cdf = histogram.cumsum()                   # Kumulatif jumlah piksel
    non_zero = cdf[cdf > 0]

    # Jika gambar sangat seragam / kosong secara logis, kembalikan CDF aman
    if non_zero.size == 0:
        return np.zeros_like(cdf, dtype=float)

    cdf_min = non_zero.min()                   # Nilai minimum non-zero
    total_pixels = cdf[-1]                    # Total jumlah piksel

    # Hindari pembagian dengan nol pada citra yang seluruh pikselnya sama
    denominator = total_pixels - cdf_min
    if denominator == 0:
        cdf_normalized = np.zeros_like(cdf, dtype=float)
        cdf_normalized[cdf >= cdf_min] = 1.0
        return cdf_normalized

    # Normalisasi CDF ke [0, 1]
    cdf_normalized = (cdf - cdf_min) / denominator
    cdf_normalized = np.clip(cdf_normalized, 0, 1)
    return cdf_normalized


def histogram_specification_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Lakukan histogram matching pada satu channel (grayscale atau satu channel RGB).

    Langkah:
    1. Hitung histogram src dan ref (masing-masing 256 bin).
    2. Hitung CDF src dan CDF ref.
    3. Untuk setiap nilai intensitas pada src, temukan nilai intensitas pada ref
       yang memiliki CDF paling dekat (look-up table / LUT).
    4. Terapkan LUT ke gambar src untuk menghasilkan gambar output.

    Args:
        src: Array 2-D piksel gambar sumber (satu channel), dtype uint8.
        ref: Array 2-D piksel gambar referensi (satu channel), dtype uint8.

    Returns:
        Array 2-D hasil histogram matching, dtype uint8.
    """
    # Hitung histogram masing-masing (256 bin, rentang 0-256)
    hist_src, _ = np.histogram(src.flatten(), bins=256, range=(0, 256))
    hist_ref, _ = np.histogram(ref.flatten(), bins=256, range=(0, 256))

    # Hitung CDF untuk kedua gambar
    cdf_src = compute_cdf(hist_src)
    cdf_ref = compute_cdf(hist_ref)

    # Buat Look-Up Table (LUT): petakan setiap intensitas src -> intensitas ref
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Cari nilai ref yang CDF-nya paling dekat dengan CDF src pada intensitas i
        diff = np.abs(cdf_ref - cdf_src[i])
        lut[i] = np.argmin(diff)

    # Terapkan LUT pada gambar sumber
    matched = lut[src]
    
    # Hitung histogram hasil untuk data langkah-langkah
    hist_res, _ = np.histogram(matched.flatten(), bins=256, range=(0, 256))
    
    steps_data = {
        # Alias disediakan agar frontend lama dan baru sama-sama kompatibel
        'hist_src': hist_src.tolist(),
        'nk_src': hist_src.tolist(),
        'cdf_src': cdf_src.tolist(),
        'hist_ref': hist_ref.tolist(),
        'nk_ref': hist_ref.tolist(),
        'cdf_ref': cdf_ref.tolist(),
        'lut': lut.tolist(),
        'hist_res': hist_res.tolist(),
        'nk_res': hist_res.tolist(),
        'n_pixels_src': int(np.sum(hist_src)),
        'n_pixels_ref': int(np.sum(hist_ref))
    }
    
    return matched, steps_data


def histogram_specification(src_img: np.ndarray, ref_img: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Lakukan histogram matching lengkap untuk gambar grayscale atau RGB.

    Jika gambar grayscale → proses langsung satu channel.
    Jika gambar RGB       → proses per-channel (R, G, B) secara terpisah.

    Args:
        src_img: Gambar sumber (NumPy array, BGR atau GRAY).
        ref_img: Gambar referensi (NumPy array, BGR atau GRAY).

    Returns:
        Gambar hasil histogram matching (format sama dengan input), dan data langkah-langkah.
    """
    if len(src_img.shape) == 2:
        # ── Gambar Grayscale ──────────────────────────────
        # Pastikan referensi juga grayscale
        if len(ref_img.shape) == 3:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        return histogram_specification_channel(src_img, ref_img)

    else:
        # ── Gambar BGR (Color) ────────────────────────────
        # Pastikan referensi juga berwarna
        if len(ref_img.shape) == 2:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

        # Proses setiap channel secara independen
        result_channels = []
        steps_data = None
        for c in range(3):  # channel 0=Blue, 1=Green, 2=Red
            matched, data = histogram_specification_channel(src_img[:, :, c], ref_img[:, :, c])
            result_channels.append(matched)
            if c == 0:
                steps_data = data  # Ambil langkah dari channel pertama sebagai representasi

        # Gabungkan kembali menjadi gambar BGR
        return cv2.merge(result_channels), steps_data


# ──────────────────────────────────────────────────────────────
#  Fungsi Visualisasi Histogram (Matplotlib)
# ──────────────────────────────────────────────────────────────

def plot_histograms(src_img: np.ndarray,
                   ref_img: np.ndarray,
                   out_img: np.ndarray) -> str:
    """
    Buat plot histogram untuk tiga gambar: sumber, referensi, dan hasil.
    Plot disimpan ke buffer lalu dikembalikan sebagai string Base64 PNG.

    Args:
        src_img: Gambar sumber.
        ref_img: Gambar referensi.
        out_img: Gambar hasil histogram matching.

    Returns:
        String Base64 gambar plot histogram.
    """
    is_gray = len(src_img.shape) == 2

    # Warna channel untuk plot BGR
    channel_colors = ['#5B8DEF', '#2ECC71', '#E74C3C']   # Blue, Green, Red
    channel_labels = ['Blue', 'Green', 'Red']

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    fig.patch.set_facecolor('#1a1a2e')

    titles = ['Histogram - Gambar Input', 'Histogram - Gambar Referensi', 'Histogram - Gambar Hasil']
    images = [src_img, ref_img, out_img]

    for ax, img, title in zip(axes, images, titles):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='#e0e0e0', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Nilai Intensitas', color='#a0a0b0', fontsize=8)
        ax.set_ylabel('Jumlah Piksel', color='#a0a0b0', fontsize=8)
        ax.tick_params(colors='#a0a0b0', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334')

        if is_gray or len(img.shape) == 2:
            # Grayscale: satu histogram putih
            gray_img = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist, bins = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))
            ax.plot(hist, color='#e0e0e0', linewidth=1.2, label='Gray')
            ax.fill_between(range(256), hist, alpha=0.25, color='#e0e0e0')
            ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#e0e0e0')
        else:
            # Color: tiga histogram per channel
            for c, (color, label) in enumerate(zip(channel_colors, channel_labels)):
                hist, _ = np.histogram(img[:, :, c].flatten(), bins=256, range=(0, 256))
                ax.plot(hist, color=color, linewidth=1.2, label=label)
                ax.fill_between(range(256), hist, alpha=0.15, color=color)
            ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#e0e0e0')

    plt.tight_layout(pad=2.0)

    # Simpan plot ke buffer memory (tidak perlu menulis file)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ──────────────────────────────────────────────────────────────
#  Route Flask
# ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Halaman utama — render template HTML."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """
    Endpoint POST untuk memproses histogram matching.

    Terima dua file gambar (input & referensi),
    lakukan histogram matching, lalu kembalikan:
      - Gambar sumber (base64)
      - Gambar referensi (base64)
      - Gambar hasil (base64)
      - Plot histogram (base64)
      sebagai JSON.
    """
    # ── Validasi Upload ───────────────────────────────────────
    if 'input_image' not in request.files or 'ref_image' not in request.files:
        return jsonify({'error': 'Harap upload kedua gambar (input dan referensi).'}), 400

    input_file = request.files['input_image']
    ref_file   = request.files['ref_image']

    if input_file.filename == '' or ref_file.filename == '':
        return jsonify({'error': 'Nama file tidak boleh kosong.'}), 400

    if not allowed_file(input_file.filename) or not allowed_file(ref_file.filename):
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG, JPG, BMP, atau TIFF.'}), 400

    # ── Baca Gambar dari Stream (tanpa menyimpan ke disk) ─────
    try:
        input_bytes = np.frombuffer(input_file.read(), dtype=np.uint8)
        ref_bytes   = np.frombuffer(ref_file.read(),   dtype=np.uint8)

        # cv2.IMREAD_UNCHANGED: jaga channel asli (gray/color)
        src_img = cv2.imdecode(input_bytes, cv2.IMREAD_UNCHANGED)
        ref_img = cv2.imdecode(ref_bytes,   cv2.IMREAD_UNCHANGED)

        if src_img is None or ref_img is None:
            return jsonify({'error': 'Gagal membaca gambar. Pastikan file tidak rusak.'}), 400

        # Jika gambar 4-channel (RGBA), konversi ke BGR
        if len(src_img.shape) == 3 and src_img.shape[2] == 4:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2BGR)
        if len(ref_img.shape) == 3 and ref_img.shape[2] == 4:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2BGR)

    except Exception as e:
        return jsonify({'error': f'Error membaca gambar: {str(e)}'}), 500

    # ── Proses Histogram Matching ─────────────────────────────
    try:
        result_img, steps_data = histogram_specification(src_img, ref_img)
    except Exception as e:
        return jsonify({'error': f'Error saat histogram matching: {str(e)}'}), 500

    # ── Buat Plot Histogram ───────────────────────────────────
    try:
        hist_plot_b64 = plot_histograms(src_img, ref_img, result_img)
    except Exception as e:
        return jsonify({'error': f'Error membuat histogram: {str(e)}'}), 500

    # ── Encode Gambar ke Base64 untuk Response ────────────────
    src_b64    = image_to_base64(src_img)
    ref_b64    = image_to_base64(ref_img)
    result_b64 = image_to_base64(result_img)

    return jsonify({
        'source_image':    src_b64,
        'ref_image':       ref_b64,
        'result_image':    result_b64,
        'histogram_plot':  hist_plot_b64,
        'is_color':        len(src_img.shape) == 3,
        'src_shape':       list(src_img.shape),
        'result_shape':    list(result_img.shape),
        'steps_data':      steps_data,
    })


# ──────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  Histogram Spesifikasi App — Flask Server")
    print("  Akses di: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host='0.0.0.0', port=5000)

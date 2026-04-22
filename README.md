# 🖼️ Computer Vision — Pertemuan 1–7

Repositori ini berisi kumpulan praktikum mata kuliah **Computer Vision** yang mencakup pemrosesan gambar dari dasar hingga implementasi jaringan saraf tiruan sederhana, menggunakan **Python**, **NumPy**, dan **Matplotlib**.

---

## 📁 Struktur Repositori

```
computer_vision/
├── github_pertemuan_1/      # Pembuatan Gambar Dasar
├── github_pertemuan_2/      # Grayscale & Normalisasi
├── github_pertemuan_3/      # Brightness & Contrast
├── github_pertemuan_5/      # Rescaling & Deteksi Kontur
└── github_pertemuan_7/      # Artificial Neural Network (ANN)
```

---

## 📚 Detail Setiap Pertemuan

---

### 🌈 Pertemuan 1 — Pembuatan Gambar dari Nol

> Dasar-dasar representasi citra digital menggunakan array NumPy.

| File | Fungsi |
|------|--------|
| `a_pelangi_horizontal.py` | Membuat gambar pelangi 5 warna dengan gradasi **horizontal** (berdasarkan baris/row). Setiap segmen baris diisi warna: Merah → Hijau → Biru → Kuning → Ungu. |
| `b_pelangi_vertikal.py` | Membuat gambar pelangi 5 warna dengan gradasi **vertikal** (berdasarkan kolom). Setiap segmen kolom diisi dengan kombinasi warna RGB yang berbeda. |
| `c_bendera_merah_putih.py` | Membuat gambar **Bendera Merah Putih** berukuran 1080×1920 px secara programatik. Bagian atas diisi warna merah, bagian bawah diisi warna putih, keduanya ditengahkan secara horizontal. |

**Konsep yang dipelajari:**
- Representasi citra sebagai array 3 dimensi `(row, col, channel)`
- Manipulasi piksel secara langsung dengan NumPy
- Visualisasi gambar menggunakan `matplotlib.pyplot.imshow()`

---

### 🌫️ Pertemuan 2 — Grayscale & Normalisasi

> Konversi citra berwarna ke skala abu-abu dan normalisasi nilai piksel.

| File | Fungsi |
|------|--------|
| `a_grayscale.py` | Mengkonversi gambar berwarna (RGB) menjadi **grayscale** menggunakan metode **Luminosity**: `gray = 0.299R + 0.587G + 0.114B`. Bobot ini merepresentasikan sensitivitas mata manusia terhadap warna. |
| `b_normalization.py` | Mengkonversi gambar RGB ke grayscale dengan metode **rata-rata** per piksel, lalu melakukan **normalisasi** nilai piksel dari rentang `[0–255]` ke rentang `[0–1]` dengan membagi setiap nilai dengan 255. |

**Konsep yang dipelajari:**
- Metode konversi grayscale (Luminosity vs. Average)
- Normalisasi nilai piksel untuk kebutuhan komputasi lebih lanjut
- Iterasi piksel secara manual menggunakan loop

---

### ☀️ Pertemuan 3 — Brightness & Contrast

> Teknik peningkatan kualitas citra dengan penyesuaian kecerahan dan kontras.

| File | Fungsi |
|------|--------|
| `a_brightness_adjusment.py` | Menyesuaikan **kecerahan (brightness)** gambar dengan menambahkan nilai `delta_Intens` ke setiap channel R, G, B setiap piksel. Nilai diklem ke minimum 0 untuk mencegah overflow negatif. |
| `b_contrast_enhancement.py` | Meningkatkan **kontras** gambar berdasarkan threshold. Piksel yang rata-rata nilainya **di atas threshold** akan dinaikkan nilainya (terang → lebih terang), dan yang **di bawah threshold** akan diturunkan (gelap → lebih gelap). Nilai diklem ke rentang `[0–255]`. |

**Konsep yang dipelajari:**
- Manipulasi kecerahan per-piksel per-channel
- Peningkatan kontras berbasis threshold
- Penggunaan `min()` dan `max()` untuk value clamping

---

### 🔍 Pertemuan 5 — Rescaling & Deteksi Kontur

> Pengubahan ukuran gambar dan deteksi tepi/kontur objek.

| File | Fungsi |
|------|--------|
| `a_rescale.py` | Mengubah ukuran (**rescale/resize**) gambar berwarna ke tinggi target (`row_res = 500 px`) secara proporsional. Menggunakan teknik **nearest-neighbor interpolation** — setiap piksel baru dipetakan ke piksel terdekat pada gambar asli berdasarkan `res_factor`. Gambar hasil disimpan ke file. |
| `b_contour_detection.py` | Mendeteksi **kontur/tepi objek** pada gambar menggunakan pendekatan berbasis **variansi lokal**. Gambar dibagi menjadi sub-image (misalnya 12×12), lalu sebuah *scanning lense* diarahkan ke tiap area untuk menghitung variansi piksel grayscale-nya. Area dengan variansi tinggi dianggap mengandung tepi. |

**Konsep yang dipelajari:**
- Algoritma resize manual dengan nearest-neighbor
- Konversi warna ke grayscale untuk analisis
- Deteksi tepi berbasis statistik (variansi lokal)
- Pembagian gambar ke sub-image untuk analisis regional

---

### 🧠 Pertemuan 7 — Artificial Neural Network (ANN)

> Implementasi jaringan saraf tiruan dari nol untuk klasifikasi bentuk geometri.

| File | Fungsi |
|------|--------|
| `b_set_dataset.py` | Menyiapkan dataset dari folder gambar bentuk geometri. Membaca semua file gambar, mendeteksi label dari nama file (circle, diamond, ellipse, dll.), mengkonversi ke grayscale 28×28, lalu **menyimpan ke file `.npz`** agar bisa dipakai ulang. |
| `c_read_dataset.py` | Modul helper untuk **memuat dataset** dari file `.npz`. Mengubah label integer ke format **one-hot encoding** (array biner) yang dibutuhkan untuk pelatihan ANN. |
| `a_ANN.py` | Implementasi lengkap **Artificial Neural Network (ANN)** 3-layer dari nol: <br>• **Input layer**: 784 neuron (28×28 piksel) <br>• **Hidden layer**: 20 neuron dengan aktivasi **Sigmoid** <br>• **Output layer**: 10 neuron (10 kelas bentuk) <br><br>Proses pelatihan menggunakan **backpropagation** manual dan **gradient descent**. Setelah training, model dapat menebak bentuk dari gambar yang diinputkan. |

**Kelas yang dapat diklasifikasikan:**
`circle`, `diamond`, `ellipse_h`, `ellipse_v`, `parallelogram`, `rectangle_h`, `rectangle_v`, `square`, `trapezium`, `triangle`

**Konsep yang dipelajari:**
- Arsitektur jaringan saraf tiruan (feedforward)
- Fungsi aktivasi Sigmoid dan turunannya
- Algoritma backpropagation dan gradient descent
- One-hot encoding untuk klasifikasi multi-kelas
- Penyimpanan dan pembacaan dataset dengan NumPy `.npz`

---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---------|----------|
| `numpy` | Operasi array, manipulasi piksel, komputasi matriks |
| `matplotlib` | Membaca, menampilkan, dan menyimpan gambar |
| `PIL (Pillow)` | Membaca gambar dataset dan konversi grayscale |

---

## ▶️ Cara Menjalankan

1. **Install dependencies:**
   ```bash
   pip install numpy matplotlib pillow
   ```

2. **Jalankan script yang diinginkan:**
   ```bash
   cd github_pertemuan_1
   python a_pelangi_horizontal.py
   ```

3. **Untuk Pertemuan 7 (ANN)**, siapkan dataset terlebih dahulu:
   ```bash
   cd github_pertemuan_7
   python b_set_dataset.py   # Buat dataset dari folder gambar
   python a_ANN.py           # Latih model dan uji hasilnya
   ```

---

## 📈 Alur Pembelajaran

```
Pertemuan 1          Pertemuan 2          Pertemuan 3
Buat Gambar    →    Grayscale &    →    Brightness &
dari Nol            Normalisasi         Contrast
                                              ↓
Pertemuan 7          Pertemuan 5
ANN Klasifikasi ←   Rescaling &
Bentuk              Deteksi Kontur
```

---

## 👨‍💻 Informasi

- **Mata Kuliah:** Computer Vision
- **Departemen:** Informatika, Universitas Pembangunan Jaya

# Pengolahan-citra-dan-video
Rafli J.S.P.T.

Untuk video demo project vtuber saya upload ke google drive, link bisa di akses disini: https://drive.google.com/file/d/1JmPYAsJ4NfIxnYz2Ua2aU78Nb_WBo6zy/view?usp=sharing

# Python VTuber Motion Capture (MediaPipe to VSeeFace) 📸➡️💃

**Python VTuber Motion Capture** adalah sistem pelacakan gerak (Motion Capture) berbasis AI yang mengubah webcam standar menjadi alat pelacak tubuh penuh (*Upper Body* + *Hands*) untuk kebutuhan VTuber.

Sistem ini menggunakan **Google MediaPipe Holistic** untuk mendeteksi wajah, pose tubuh, dan tangan, kemudian mengonversi data tersebut menjadi protokol VMC (Virtual Motion Capture) dan mengirimkannya ke aplikasi **VSeeFace** melalui protokol OSC (UDP).

![Python to VSeeFace Visualization](https://via.placeholder.com/800x400?text=Screenshot+Visualisasi+Tracking+Anda)
*(Ganti gambar di atas dengan screenshot hasil tracking program Anda)*

## 🌟 Fitur Utama

* **Full Upper Body Tracking:** Melacak rotasi kepala, bahu, siku, dan pergelangan tangan dengan koreksi rotasi tulang (Quaternion).
* **High Precision Hand Tracking:** Deteksi pergerakan 10 jari secara individual.
* **Advanced Mouth Tracking (MAR):** Menggunakan *Mouth Aspect Ratio* (MAR) sehingga bukaan mulut tetap akurat meskipun jarak wajah ke kamera berubah-ubah (maju/mundur).
* **Smart Smoothing:** Menggunakan **One Euro Filter** untuk menghilangkan getaran (jitter) saat diam, namun tetap responsif saat bergerak cepat.
* **Visualisasi Debugging:** Dilengkapi GUI yang menampilkan:
    * Jaring wajah (Tesselation).
    * Indikator kepercayaan tracking (Titik Hijau/Merah di bahu).
    * Garis bantu deteksi mulut (Kuning).
* **Eye Tracking:** Deteksi kedipan dan pergerakan iris mata.

## 🛠️ Prasyarat (Requirements)

Pastikan Anda telah menginstal **Python 3.8** atau versi yang lebih baru. Proyek ini membutuhkan pustaka berikut:

* `opencv-python` (Pengolahan citra)
* `mediapipe` (AI Tracking)
* `python-osc` (Komunikasi ke VSeeFace)
* `numpy` (Kalkulasi Matematika)

## 📦 Instalasi

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
    cd nama-repo-anda
    ```

2.  **Instal dependensi:**
    Salin dan jalankan perintah ini di terminal/CMD:
    ```bash
    pip install opencv-python mediapipe python-osc numpy
    ```

## ⚙️ Konfigurasi VSeeFace

Agar karakter VTuber Anda bergerak mengikuti data dari Python, lakukan pengaturan berikut di **VSeeFace**:

1.  Buka VSeeFace dan muat avatar Anda.
2.  Masuk ke **Settings** > **General Settings**.
3.  Cari bagian **OSC / VMC Receiver**.
4.  Centang **Enable**.
5.  Pastikan konfigurasi berikut sesuai:
    * **Port:** `39539` (Default)
    * **IP Address:** `127.0.0.1` (Localhost)
6.  **Penting:** Pada bagian pemilihan kamera di VSeeFace, pilih **(None)** atau matikan kamera bawaan VSeeFace agar tidak bentrok dengan skrip Python.

## 🚀 Cara Penggunaan

1.  Sambungkan webcam ke komputer.
2.  Jalankan skrip utama:
    ```bash
    python vseeface_sender.py
    ```
    *(Sesuaikan nama file jika Anda mengubahnya)*
3.  Akan muncul jendela visualisasi.
    * **Garis Kuning di Mulut:** Menandakan deteksi bibir aktif.
    * **Titik Hijau di Bahu:** Menandakan tracking tubuh stabil.
    * **Titik Merah:** Menandakan tracking hilang (sistem akan menahan gerakan agar avatar tidak *glitch*).
4.  Buka VSeeFace, dan avatar Anda akan mulai bergerak!
5.  Tekan tombol **`q`** pada keyboard di jendela Python untuk keluar.

## 🔧 Kustomisasi & Tuning

Anda dapat mengubah variabel di bagian atas kode (`vseeface_sender.py`) untuk menyesuaikan sensitivitas:

### 1. Sensitivitas Mulut
Jika mulut avatar tidak mau menutup rapat atau tidak terbuka lebar:
```python
MOUTH_MIN_RATIO = 0.05  # Rasio saat mulut diam/tutup
MOUTH_MAX_RATIO = 0.35  # Rasio saat mulut terbuka lebar ('A')

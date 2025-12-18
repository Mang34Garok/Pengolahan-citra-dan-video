import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math
import time

# ==============================================================================
# 1. KONFIGURASI & PENGATURAN
# ==============================================================================
# Konfigurasi Koneksi ke VSeeFace
OSC_IP = "127.0.0.1"   # Alamat IP (Localhost = komputer ini sendiri)
OSC_PORT = 39539       # Port standar VSeeFace (bisa dicek di General Settings VSeeFace)
WEBCAM_ID = 0          # ID Kamera (0 = Default/Webcam Laptop)
TARGET_FPS = 30        # Target kecepatan frame per detik

# Kalibrasi Sensitivitas Mulut
# Ubah ini jika mulut avatar tidak mau menutup rapat atau membuka lebar
MOUTH_MIN_RATIO = 0.05 # Rasio saat mulut diam (tutup)
MOUTH_MAX_RATIO = 0.35 # Rasio saat mulut terbuka lebar ('A')

# Batas Kepercayaan AI (Confidence Threshold)
# Jika AI yakin di bawah 60% (0.6), posisi badan tidak akan diupdate (untuk mencegah glitch)
CONFIDENCE_THRESHOLD = 0.6 

# Pengaturan Smoothing (Filter Penghalus Gerakan)
# min_cutoff kecil = Gerakan lambat sangat halus. beta besar = Gerakan cepat responsif.
HEAD_MIN_CUTOFF = 0.05 ; HEAD_BETA = 0.5   # Untuk Kepala
BODY_MIN_CUTOFF = 0.05 ; BODY_BETA = 0.5   # Untuk Tulang Belakang
ARM_MIN_CUTOFF  = 0.1  ; ARM_BETA  = 0.6   # Untuk Lengan (perlu responsif)
FINGER_MIN_CUTOFF = 0.5; FINGER_BETA = 1.0 # Untuk Jari (harus cepat)

# ==============================================================================
# 2. FUNGSI-FUNGSI MATEMATIKA (HELPER)
# ==============================================================================

def euler_to_quaternion(pitch, yaw, roll):
    """
    Mengubah sudut derajat (Euler) menjadi Quaternion (x,y,z,w).
    VSeeFace/Unity menggunakan Quaternion agar rotasi 3D tidak 'terkunci' (Gimbal Lock).
    """
    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    qy = np.cos(pitch/2) * np.sin(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.cos(yaw/2) * np.sin(roll/2)
    qz = np.cos(pitch/2) * np.cos(yaw/2) * np.sin(roll/2) - np.sin(pitch/2) * np.sin(yaw/2) * np.cos(roll/2)
    qw = np.cos(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    return [qx, qy, qz, qw]

def get_limb_rotation(start, end, rest_vector):
    """
    Fungsi Paling Penting untuk Badan!
    Menghitung rotasi tulang lengan berdasarkan vektor posisi awal (bahu) dan akhir (siku).
    Membandingkan posisi saat ini dengan posisi istirahat (T-Pose).
    """
    v_curr = np.array(end) - np.array(start) # Vektor tulang saat ini
    norm = np.linalg.norm(v_curr)
    if norm < 1e-6: return [0,0,0,1] # Hindari error pembagian nol
    
    v_curr = v_curr / norm # Normalisasi vektor
    v_rest = np.array(rest_vector) # Vektor T-Pose
    v_rest = v_rest / np.linalg.norm(v_rest)
    
    # Hitung sudut perbedaan
    dot = np.dot(v_rest, v_curr)
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)
    
    # Hitung sumbu putar
    axis = np.cross(v_rest, v_curr)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6: return [0, 0, 0, 1]
    
    axis = axis / axis_len
    # Konversi ke Quaternion
    sin_half = math.sin(angle / 2)
    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = math.cos(angle / 2)
    return [qx, qy, qz, qw]

def get_finger_curl(landmarks, tip_idx, knuckle_idx, wrist_idx):
    """Mendeteksi seberapa menekuk jari (0.0 lurus - 1.0 menekuk penuh)"""
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y])
    dist_tip_wrist = np.linalg.norm(tip - wrist)
    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y])
    dist_palm = np.linalg.norm(knuckle - wrist)
    # Rumus empiris rasio jarak
    return (dist_tip_wrist / (dist_palm + 1e-6) - 1.9) / (0.8 - 1.9)

def get_finger_quat(angle, axis_idx):
    """Menghitung rotasi tulang jari berdasarkan sudut tekukan"""
    s, c = math.sin(angle/2), math.cos(angle/2)
    # axis_idx: 1=Tekuk Depan (Thumb), 2=Tekuk Bawah (Jari lain)
    if axis_idx == 0: return [s, 0, 0, c]
    elif axis_idx == 1: return [0, s, 0, c]
    elif axis_idx == 2: return [0, 0, s, c]
    return [0, 0, 0, 1]

def calculate_ear(fl, indices, w, h):
    """Menghitung EAR (Eye Aspect Ratio) untuk deteksi kedipan mata"""
    coords = [np.array([fl.landmark[i].x*w, fl.landmark[i].y*h]) for i in indices]
    v1 = np.linalg.norm(coords[1]-coords[5]) # Tinggi mata bagian 1
    v2 = np.linalg.norm(coords[2]-coords[4]) # Tinggi mata bagian 2
    return (v1 + v2) / (2.0 * np.linalg.norm(coords[0]-coords[3]) + 1e-6) # Dibagi lebar mata

def calculate_mar(fl, w, h):
    """Menghitung MAR (Mouth Aspect Ratio) untuk bukaan mulut"""
    top = np.array([fl.landmark[13].x*w, fl.landmark[13].y*h])   # Bibir Atas
    bot = np.array([fl.landmark[14].x*w, fl.landmark[14].y*h])   # Bibir Bawah
    left = np.array([fl.landmark[61].x*w, fl.landmark[61].y*h])  # Ujung Kiri
    right = np.array([fl.landmark[291].x*w, fl.landmark[291].y*h]) # Ujung Kanan
    # Rasio = Tinggi / Lebar
    return np.linalg.norm(top-bot)/(np.linalg.norm(left-right)+1e-6), top, bot, left, right

# ==============================================================================
# 3. CLASS ONE EURO FILTER (ALGORITMA SMOOTHING)
# ==============================================================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x_prev, self.dx_prev, self.t_prev = float(x0), float(dx0), float(t0)

    def __call__(self, t, x):
        """Memproses nilai baru (x) pada waktu (t) untuk dihaluskan"""
        t_e = t - self.t_prev
        if t_e <= 0.0: return self.x_prev
        # Hitung kecepatan perubahan (derivative)
        a_d = 2*math.pi*self.d_cutoff*t_e / (2*math.pi*self.d_cutoff*t_e + 1)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d*dx + (1-a_d)*self.dx_prev
        # Hitung cutoff frequency secara adaptif
        cutoff = self.min_cutoff + self.beta*abs(dx_hat)
        a = 2*math.pi*cutoff*t_e / (2*math.pi*cutoff*t_e + 1)
        x_hat = a*x + (1-a)*self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

# ==============================================================================
# 4. INISIALISASI VARIABEL & OBJEK
# ==============================================================================
# Siapkan MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Model Complexity 1 = Seimbang antara cepat dan akurat
holistic = mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6, refine_face_landmarks=True, model_complexity=1)

# Siapkan Client OSC untuk mengirim data
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# Inisialisasi Filter untuk setiap bagian tubuh (agar gerakan halus)
t_start = time.time()
filter_pitch = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_yaw   = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_roll  = OneEuroFilter(t_start, 0, min_cutoff=HEAD_MIN_CUTOFF, beta=HEAD_BETA)
filter_spine_roll = OneEuroFilter(t_start, 0, min_cutoff=BODY_MIN_CUTOFF, beta=BODY_BETA)
filter_spine_yaw  = OneEuroFilter(t_start, 0, min_cutoff=BODY_MIN_CUTOFF, beta=BODY_BETA)

# Filter Lengan (Bahu, Siku, Pergelangan) - Masing-masing punya filter X, Y, Z
filter_l_sh = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_l_el = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_l_wr = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_r_sh = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_r_el = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]
filter_r_wr = [OneEuroFilter(t_start, 0, min_cutoff=ARM_MIN_CUTOFF, beta=ARM_BETA) for _ in range(3)]

# Filter Jari & Mulut
filters_fingers_L = [OneEuroFilter(t_start, 0, min_cutoff=FINGER_MIN_CUTOFF, beta=FINGER_BETA) for _ in range(5)]
filters_fingers_R = [OneEuroFilter(t_start, 0, min_cutoff=FINGER_MIN_CUTOFF, beta=FINGER_BETA) for _ in range(5)]
filter_mouth = OneEuroFilter(t_start, 0, min_cutoff=0.01, beta=2.0)

# Data Referensi Wajah 3D (untuk PnP Solver)
model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
# Index landmark mata & jari
LEFT_EYE_IDXS, RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_INDICES = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
prev_time = 0

# Buka Kamera
cap = cv2.VideoCapture(WEBCAM_ID)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

print("=== VTuber SYSTEM (Full Body) STARTED ===")
print("Tekan 'q' untuk keluar.")

# ==============================================================================
# 5. LOOP UTAMA (PROGRAM BERJALAN DI SINI)
# ==============================================================================
while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    # Hitung FPS Realtime
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 and (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Proses Gambar
    image = cv2.flip(image, 1) # Mirror image agar natural
    h, w, _ = image.shape
    
    image.flags.writeable = False
    # --- PROSES MEDIAPIPE (DETEKSI) ---
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.flags.writeable = True

    # -----------------------------------------------------------
    # BAGIAN A: VISUALISASI (GAMBAR GARIS DI LAYAR)
    # -----------------------------------------------------------
    if results.face_landmarks:
        # Gambar jaring wajah
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())
        # Gambar titik merah di pupil mata
        if results.face_landmarks.landmark:
             lm = results.face_landmarks.landmark
             cv2.circle(image, (int(lm[468].x*w), int(lm[468].y*h)), 3, (0,0,255), -1)
             cv2.circle(image, (int(lm[473].x*w), int(lm[473].y*h)), 3, (0,0,255), -1)

    if results.pose_landmarks:
        # Gambar tulang badan
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing_styles.get_default_pose_landmarks_style())

    # Gambar tulang tangan
    if results.left_hand_landmarks: mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks: mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # -----------------------------------------------------------
    # BAGIAN B: LOGIKA WAJAH (KEPALA, MATA, MULUT)
    # -----------------------------------------------------------
    if results.face_landmarks:
        fl = results.face_landmarks
        
        # --- 1. MULUT (MAR Calculation) ---
        raw_mar, pt_top, pt_bot, pt_left, pt_right = calculate_mar(fl, w, h)
        smooth_mar = filter_mouth(curr_time, raw_mar)
        # Mapping nilai MAR ke nilai 0.0-1.0
        mouth_open = max(0.0, min(1.0, (smooth_mar - MOUTH_MIN_RATIO)/(MOUTH_MAX_RATIO - MOUTH_MIN_RATIO)))
        
        # Visualisasi Garis Mulut (Kuning)
        cv2.line(image, (int(pt_top[0]), int(pt_top[1])), (int(pt_bot[0]), int(pt_bot[1])), (0, 255, 255), 2)
        cv2.line(image, (int(pt_left[0]), int(pt_left[1])), (int(pt_right[0]), int(pt_right[1])), (0, 255, 255), 2)
        # Kirim data mulut ke VSeeFace
        client.send_message("/VMC/Ext/Blend/Val", ["A", float(mouth_open)])

        # --- 2. KEPALA (PnP Solver) ---
        # Mengambil 6 titik wajah untuk estimasi rotasi 3D
        image_points = np.array([
            (fl.landmark[1].x*w, fl.landmark[1].y*h), (fl.landmark[152].x*w, fl.landmark[152].y*h),
            (fl.landmark[263].x*w, fl.landmark[263].y*h), (fl.landmark[33].x*w, fl.landmark[33].y*h),
            (fl.landmark[287].x*w, fl.landmark[287].y*h), (fl.landmark[57].x*w, fl.landmark[57].y*h)
        ], dtype=np.float64)
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
        
        # Algoritma SolvePnP untuk menghitung rotasi
        success, rot_vec, _ = cv2.solvePnP(model_points, image_points, cam_matrix, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Filter hasil rotasi agar tidak gemetar
        s_pitch = filter_pitch(curr_time, angles[0])
        s_yaw   = filter_yaw(curr_time, angles[1])
        # Hitung Roll manual dari sudut mata
        s_roll  = filter_roll(curr_time, math.degrees(math.atan2((fl.landmark[263].y*h)-(fl.landmark[33].y*h), (fl.landmark[263].x*w)-(fl.landmark[33].x*w))))

        # Kirim Data Rotasi Kepala & Leher
        nqx, nqy, nqz, nqw = euler_to_quaternion(math.radians(s_pitch*0.5), math.radians(s_yaw*0.5), math.radians(s_roll*0.5))
        hqx, hqy, hqz, hqw = euler_to_quaternion(math.radians(s_pitch*0.5), math.radians(s_yaw*0.5), math.radians(s_roll*0.5))
        client.send_message("/VMC/Ext/Bone/Pos", ["Neck", 0.0, 0.0, 0.0, float(nqx), float(nqy), float(nqz), float(nqw)])
        client.send_message("/VMC/Ext/Bone/Pos", ["Head", 0.0, 0.0, 0.0, float(hqx), float(hqy), float(hqz), float(hqw)])
        
        # --- 3. MATA (Kedipan) ---
        blink_l = 1.0 if calculate_ear(fl, LEFT_EYE_IDXS, w, h) < 0.15 else 0.0
        blink_r = 1.0 if calculate_ear(fl, RIGHT_EYE_IDXS, w, h) < 0.15 else 0.0
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_L", float(blink_l)])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_R", float(blink_r)])

    # -----------------------------------------------------------
    # BAGIAN C: LOGIKA BADAN & LENGAN
    # -----------------------------------------------------------
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Fungsi Lokal: Memfilter posisi X,Y,Z bahu/siku
        def get_filtered_vec(idx, filters):
            raw = [lm[idx].x, lm[idx].y, lm[idx].z]
            return [filters[0](curr_time, raw[0]), filters[1](curr_time, raw[1]), filters[2](curr_time, raw[2])], lm[idx].visibility
        
        # Fungsi Lokal: Kalibrasi koordinat ke Unity
        def to_unity(vec): return np.array([vec[0]*1.2, vec[1]*1.2, vec[2]*0.4])

        # Ambil Koordinat Bahu, Siku, Pergelangan (Kiri & Kanan)
        l_sh, l_sh_v = get_filtered_vec(11, filter_l_sh)
        l_el, l_el_v = get_filtered_vec(13, filter_l_el)
        l_wr, l_wr_v = get_filtered_vec(15, filter_l_wr)
        r_sh, r_sh_v = get_filtered_vec(12, filter_r_sh)
        r_el, r_el_v = get_filtered_vec(14, filter_r_el)
        r_wr, r_wr_v = get_filtered_vec(16, filter_r_wr)

        # Visualisasi Confidence (Titik Hijau = Bagus, Merah = Hilang)
        cv2.circle(image, (int(l_sh[0]*w), int(l_sh[1]*h)), 8, (0,255,0) if l_sh_v > CONFIDENCE_THRESHOLD else (0,0,255), -1)
        cv2.circle(image, (int(r_sh[0]*w), int(r_sh[1]*h)), 8, (0,255,0) if r_sh_v > CONFIDENCE_THRESHOLD else (0,0,255), -1)

        # --- 1. SPINE (TULANG BELAKANG) ---
        # Hanya update jika kedua bahu terlihat jelas
        if l_sh_v > CONFIDENCE_THRESHOLD and r_sh_v > CONFIDENCE_THRESHOLD:
            # Rotasi Spine berdasarkan kemiringan bahu
            spine_roll = (l_sh[1] - r_sh[1]) * -120.0
            spine_yaw  = (l_sh[2] - r_sh[2]) * -80.0
            sqx, sqy, sqz, sqw = euler_to_quaternion(0, math.radians(spine_yaw), math.radians(spine_roll))
            client.send_message("/VMC/Ext/Bone/Pos", ["Spine", 0.0, 0.0, 0.0, float(sqx), float(sqy), float(sqz), float(sqw)])

        # --- 2. LENGAN KIRI ---
        if l_sh_v > CONFIDENCE_THRESHOLD and l_el_v > CONFIDENCE_THRESHOLD:
            start, end = to_unity(l_sh), to_unity(l_el)
            # Hitung rotasi bahu kiri (relatif thd sumbu X)
            q_lu = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
            client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperArm", 0.0, 0.0, 0.0, float(q_lu[0]), float(q_lu[1]), float(q_lu[2]), float(q_lu[3])])
            
            # Hitung rotasi siku kiri
            if l_wr_v > CONFIDENCE_THRESHOLD:
                start, end = to_unity(l_el), to_unity(l_wr)
                q_ll = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerArm", 0.0, 0.0, 0.0, float(q_ll[0]), float(q_ll[1]), float(q_ll[2]), float(q_ll[3])])

        # --- 3. LENGAN KANAN ---
        if r_sh_v > CONFIDENCE_THRESHOLD and r_el_v > CONFIDENCE_THRESHOLD:
            start, end = to_unity(r_sh), to_unity(r_el)
            # Hitung rotasi bahu kanan (relatif thd sumbu -X)
            q_ru = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
            client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperArm", 0.0, 0.0, 0.0, float(q_ru[0]), float(q_ru[1]), float(q_ru[2]), float(q_ru[3])])
            
            # Hitung rotasi siku kanan
            if r_wr_v > CONFIDENCE_THRESHOLD:
                start, end = to_unity(r_el), to_unity(r_wr)
                q_rl = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerArm", 0.0, 0.0, 0.0, float(q_rl[0]), float(q_rl[1]), float(q_rl[2]), float(q_rl[3])])

    # -----------------------------------------------------------
    # BAGIAN D: LOGIKA JARI
    # -----------------------------------------------------------
    if results.left_hand_landmarks:
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            # Hitung tekukan jari
            curl = filters_fingers_L[i](curr_time, get_finger_curl(results.left_hand_landmarks, tip, knuckle, 0))
            # Tentukan sudut dan sumbu putar (Jempol beda sumbu dengan jari lain)
            angle = curl * (math.pi/2) * -1.0 if name == "Thumb" else curl * (math.pi/1.5)
            fqx, fqy, fqz, fqw = get_finger_quat(angle, 1 if name == "Thumb" else 2)
            # Kirim data ke tulang jari
            for s in ["Proximal", "Intermediate"]: client.send_message(f"/VMC/Ext/Bone/Pos", [f"Left{name}{s}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    if results.right_hand_landmarks:
        for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            curl = filters_fingers_R[i](curr_time, get_finger_curl(results.right_hand_landmarks, tip, knuckle, 0))
            angle = curl * (math.pi/2) * 1.0 if name == "Thumb" else curl * (math.pi/1.5) * -1.0
            fqx, fqy, fqz, fqw = get_finger_quat(angle, 1 if name == "Thumb" else 2)
            for s in ["Proximal", "Intermediate"]: client.send_message(f"/VMC/Ext/Bone/Pos", [f"Right{name}{s}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    # Tampilkan FPS dan Gambar Akhir
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('VTuber Full Body + Visuals', image)
    
    # Tekan 'q' untuk keluar loop
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Bersihkan memori dan tutup kamera
cap.release()
cv2.destroyAllWindows()
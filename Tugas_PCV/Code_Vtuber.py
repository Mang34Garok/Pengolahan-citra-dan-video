import cv2
import mediapipe as mp
import numpy as np
import math

# --- Inisialisasi MediaPipe ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

cap = cv2.VideoCapture(0)

# --- Konstanta Warna (BGR) ---
COLOR_BG = (240, 240, 240)      # Abu-abu muda
COLOR_SKIN = (180, 200, 255)    # Warna kulit
COLOR_EYE_WHITE = (255, 255, 255)
COLOR_PUPIL = (50, 50, 50)
COLOR_MOUTH = (50, 50, 200)     
COLOR_HAND = (180, 200, 255)    
COLOR_ARM = (150, 170, 220)     

# --- Indeks Landmark ---
LIP_TOP = 13
LIP_BOTTOM = 14
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# --- Fungsi Bantuan ---
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def denormalize_coords(landmark, img_w, img_h):
    return int(landmark.x * img_w), int(landmark.y * img_h)

# --- FUNGSI UTAMA PENGGAMBAR AVATAR ---
def draw_dynamic_avatar(canvas, nose_pos, ls_pos, rs_pos, scale, mouth_ratio, blink_l, blink_r, l_hand_pos, r_hand_pos):
    """
    ls_pos: Posisi Bahu Kiri (Left Shoulder)
    rs_pos: Posisi Bahu Kanan (Right Shoulder)
    """
    cx, cy = nose_pos
    
    # 1. Gambar Badan (Segitiga tumpul dari bahu ke bawah)
    # Ini membuat badan mengikuti kemiringan bahu asli
    body_low_x = cx
    body_low_y = cy + int(200 * scale)
    triangle_cnt = np.array([ls_pos, rs_pos, (body_low_x, body_low_y)], np.int32)
    cv2.drawContours(canvas, [triangle_cnt], 0, COLOR_SKIN, -1)
    
    # Tambah lingkaran di setiap bahu agar sendi terlihat mulus
    shoulder_radius = int(25 * scale)
    cv2.circle(canvas, ls_pos, shoulder_radius, COLOR_SKIN, -1)
    cv2.circle(canvas, rs_pos, shoulder_radius, COLOR_SKIN, -1)

    # 2. Gambar Kepala (Lingkaran)
    head_radius = int(80 * scale)
    cv2.circle(canvas, (cx, cy), head_radius, COLOR_SKIN, -1)

    # --- MATA ---
    eye_offset_x = int(35 * scale)
    eye_offset_y = int(10 * scale)
    eye_radius = int(20 * scale)
    
    # Mata Kiri (Screen Left)
    lx, ly = cx - eye_offset_x, cy - eye_offset_y
    if blink_l:
        cv2.line(canvas, (lx - eye_radius, ly), (lx + eye_radius, ly), COLOR_PUPIL, int(4*scale))
    else:
        cv2.circle(canvas, (lx, ly), eye_radius, COLOR_EYE_WHITE, -1)
        cv2.circle(canvas, (lx, ly), int(8*scale), COLOR_PUPIL, -1)

    # Mata Kanan (Screen Right)
    rx, ry = cx + eye_offset_x, cy - eye_offset_y
    if blink_r:
        cv2.line(canvas, (rx - eye_radius, ry), (rx + eye_radius, ry), COLOR_PUPIL, int(4*scale))
    else:
        cv2.circle(canvas, (rx, ry), eye_radius, COLOR_EYE_WHITE, -1)
        cv2.circle(canvas, (rx, ry), int(8*scale), COLOR_PUPIL, -1)

    # --- MULUT ---
    mouth_y = cy + int(35 * scale)
    mouth_w = int(40 * scale)
    mouth_h = max(int(5 * scale), int(mouth_ratio * 120 * scale)) 
    
    if mouth_h <= int(8 * scale):
        cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, int(15*scale)), 0, 0, 180, COLOR_MOUTH, int(4*scale))
    else:
        cv2.ellipse(canvas, (cx, mouth_y + mouth_h//2), (mouth_w, mouth_h), 0, 0, 360, COLOR_MOUTH, -1)

    # --- TANGAN (PERBAIKAN LOGIKA) ---
    hand_radius = int(25 * scale)
    arm_thickness = int(12 * scale)
    
    # Tangan Kiri (Screen Left) -> Hubungkan ke Bahu Kiri (ls_pos)
    if l_hand_pos:
        cv2.line(canvas, ls_pos, l_hand_pos, COLOR_ARM, arm_thickness)
        cv2.circle(canvas, l_hand_pos, hand_radius, COLOR_HAND, -1)
        
    # Tangan Kanan (Screen Right) -> Hubungkan ke Bahu Kanan (rs_pos)
    if r_hand_pos:
        cv2.line(canvas, rs_pos, r_hand_pos, COLOR_ARM, arm_thickness)
        cv2.circle(canvas, r_hand_pos, hand_radius, COLOR_HAND, -1)


# --- Loop Utama ---
while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Flip image AGAR SEPERTI CERMIN
    image = cv2.flip(image, 1)
    h_img, w_img, _ = image.shape
    
    avatar_canvas = np.full((h_img, w_img, 3), COLOR_BG, dtype=np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # --- Default Values ---
    nose_px = (w_img // 2, h_img // 3)
    # Default bahu jika tidak terdeteksi (di kiri dan kanan hidung)
    ls_px = (nose_px[0] - 100, nose_px[1] + 100)
    rs_px = (nose_px[0] + 100, nose_px[1] + 100)
    
    scale = 1.0
    mouth_ratio = 0.0
    blink_l, blink_r = False, False
    pos_hand_l, pos_hand_r = None, None

    # 1. POSE (BADAN & BAHU)
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        nose = lm[mp_holistic.PoseLandmark.NOSE]
        left_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_sh = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

        if left_sh.visibility > 0.5 and right_sh.visibility > 0.5:
            # Konversi ke pixel
            nose_px = denormalize_coords(nose, w_img, h_img)
            ls_px = denormalize_coords(left_sh, w_img, h_img)
            rs_px = denormalize_coords(right_sh, w_img, h_img)
            
            # Skala berdasarkan lebar bahu
            shoulder_width = calculate_distance(left_sh, right_sh)
            scale = shoulder_width * 1.8 # Sedikit diperbesar

    # 2. WAJAH (MATA & MULUT)
    if results.face_landmarks:
        mesh = results.face_landmarks.landmark
        
        # Mulut
        m_dist = calculate_distance(mesh[LIP_TOP], mesh[LIP_BOTTOM])
        mouth_ratio = max(0, (m_dist / scale) - 0.02) if scale > 0 else 0
        
        # Kedipan
        BLINK_TH = 0.012
        if calculate_distance(mesh[LEFT_EYE_TOP], mesh[LEFT_EYE_BOTTOM]) < BLINK_TH: blink_l = True
        if calculate_distance(mesh[RIGHT_EYE_TOP], mesh[RIGHT_EYE_BOTTOM]) < BLINK_TH: blink_r = True

    # 3. TANGAN (PERBAIKAN MAPPING)
    # Karena kita sudah FLIP gambar di awal, "Left Hand" MediaPipe ada di KIRI layar.
    # Maka kita hubungkan ke "ls_px" (Bahu Kiri di layar).
    
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        pos_hand_l = denormalize_coords(wrist, w_img, h_img)
        
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        pos_hand_r = denormalize_coords(wrist, w_img, h_img)

    # --- GAMBAR ---
    draw_dynamic_avatar(
        avatar_canvas, 
        nose_px, ls_px, rs_px, # Kirim posisi bahu asli
        scale, 
        mouth_ratio, 
        blink_l, blink_r, 
        pos_hand_l, pos_hand_r
    )

    # Tampilkan overlay kamera asli
    img_sm = cv2.resize(image, (w_img // 4, h_img // 4))
    avatar_canvas[h_img - h_img//4 - 10 : h_img - 10, 10 : 10 + w_img//4] = img_sm

    cv2.imshow('Avatar Fixed Hand Position', avatar_canvas)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
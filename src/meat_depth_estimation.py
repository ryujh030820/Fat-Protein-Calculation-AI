import torch
import cv2
import mediapipe as mp
import numpy as np

# 모델 로드
model_type = "DPT_Large"  # 사용할 모델 유형 선택 ("DPT_Large", "DPT_Hybrid", "MiDaS_small" 등)
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# 장치 설정 (GPU 사용 가능 여부 확인)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# 이미지 전처리를 위한 변환 로드
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# 입력 이미지 불러오기
img = cv2.imread("testpng/test.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지 변환 및 텐서로 변환
input_batch = transform(img_rgb).to(device)

# 깊이 예측
with torch.no_grad():
    prediction = midas(input_batch)

    # 원본 이미지 크기로 리사이즈
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# 결과를 NumPy 배열로 변환
depth_map = prediction.cpu().numpy()

# 깊이 맵을 시각화하기 위해 0-255 범위로 정규화
depth_min = depth_map.min()
depth_max = depth_map.max()
depth_map = (depth_map - depth_min) / (depth_max - depth_min)
depth_map = (depth_map * 255).astype(np.uint8)

# 깊이 맵 저장
cv2.imwrite("depth_map.png", depth_map)

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Mediapipe를 사용하여 손 랜드마크 검출
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    result = hands.process(img_rgb)

# 손바닥 길이 계산
if result.multi_hand_landmarks:
    hand_landmarks = result.multi_hand_landmarks[0]

    # 손바닥 길이를 계산하기 위한 랜드마크 선택 (랜드마크 0과 9)
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # 이미지 크기에 맞게 좌표 변환
    h, w, _ = img.shape
    wrist_coords = (int(wrist.x * w), int(wrist.y * h))
    middle_mcp_coords = (int(middle_mcp.x * w), int(middle_mcp.y * h))

    # 손바닥 길이 계산 (픽셀 단위)
    palm_length_px = np.linalg.norm(np.array(wrist_coords) - np.array(middle_mcp_coords))

    # 손바닥의 실제 길이 (밀리미터 단위) - 사용자의 실제 손바닥 길이를 입력하거나 평균 값을 사용
    actual_palm_length_mm = 180  # 예를 들어 100mm로 설정

    # 픽셀 당 실제 거리 계산 (mm/pixel)
    mm_per_pixel = actual_palm_length_mm / palm_length_px

    print(f"손바닥 길이 (픽셀): {palm_length_px}")
    print(f"픽셀 당 실제 거리 (mm): {mm_per_pixel}")

    # 손 랜드마크를 이미지에 그리기 (옵션)
    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imwrite('hand_landmarks.png', img)
else:
    print("손을 검출하지 못했습니다.")
    exit()

# 고기 마스크 이미지 불러오기
meat_mask = cv2.imread('testpng/maskedtest.png', cv2.IMREAD_GRAYSCALE)
if meat_mask is None:
    print("고기 마스크 이미지를 불러오지 못했습니다.")
    exit()

# 깊이 맵 불러오기
depth_map = cv2.imread('depth_map.png', cv2.IMREAD_GRAYSCALE)
if depth_map is None:
    print("깊이 맵 이미지를 불러오지 못했습니다.")
    exit()

# 고기 영역의 깊이 값 추출
meat_depth_values = depth_map[meat_mask > 0]

# 고기 두께 계산 (깊이 값의 최대값과 최소값의 차이)
depth_max = np.max(meat_depth_values)
depth_min = np.min(meat_depth_values)
meat_thickness_px = depth_max - depth_min

# 고기 두께를 실제 거리로 변환 (밀리미터 단위)
meat_thickness_mm = meat_thickness_px * mm_per_pixel

print(f"고기 두께 (픽셀): {meat_thickness_px}")
print(f"고기 두께 (밀리미터): {meat_thickness_mm:.2f} mm")
import torch
import cv2
import mediapipe as mp
import numpy as np

# 모델 로드
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# 장치 설정
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# 이미지 전처리를 위한 변환 로드
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# 입력 이미지 읽기
img = cv2.imread("testpng/test.jpeg")
if img is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)

# 깊이 예측
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
    ).squeeze()

# 깊이 맵 변환
depth_map = prediction.cpu().numpy()
depth_min, depth_max = depth_map.min(), depth_map.max()
depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

cv2.imwrite("depth_map.png", depth_map)

# Mediapipe Hands 초기화
mp_hands = mp.solutions.hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
    results = hands.process(img_rgb)

if results.multi_hand_landmarks:
    hand_landmarks = results.multi_hand_landmarks[0]
    h, w, _ = img.shape
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    wrist_coords = (int(wrist.x * w), int(wrist.y * h))
    middle_mcp_coords = (int(middle_mcp.x * w), int(middle_mcp.y * h))
    palm_length_px = np.linalg.norm(np.array(wrist_coords) - np.array(middle_mcp_coords))
    actual_palm_length_mm = 180  # 실제 손바닥 길이 (예: 180mm)
    mm_per_pixel = actual_palm_length_mm / palm_length_px
else:
    print("손을 검출하지 못했습니다.")
    exit()

# 고기 마스크 이미지 읽기
meat_mask = cv2.imread('testpng/maskedtest.png', cv2.IMREAD_GRAYSCALE)
if meat_mask is None:
    print("고기 마스크 이미지를 불러오지 못했습니다.")
    exit()

# 깊이 맵 읽기
depth_map = cv2.imread('depth_map.png', cv2.IMREAD_GRAYSCALE)
if depth_map is None:
    print("깊이 맵 이미지를 불러오지 못했습니다.")
    exit()

# 고기 영역의 깊이 값 추출
meat_depth_values = depth_map[meat_mask > 0]
depth_max, depth_min = np.max(meat_depth_values), np.min(meat_depth_values)
meat_thickness_px = depth_max - depth_min
meat_thickness_cm = meat_thickness_px * mm_per_pixel / 10  # mm -> cm 변환
print(f"고기 두께 (cm): {meat_thickness_cm:.2f}")

# 1cm²당 픽셀 수 계산
def calculate_pixels_per_cm2(palm_length_pixels, palm_length_cm):
    pixels_per_cm = palm_length_pixels / palm_length_cm
    return pixels_per_cm ** 2

pixels_per_cm2 = calculate_pixels_per_cm2(palm_length_px, actual_palm_length_mm / 10)

# 이미지에서 검은색 제외한 픽셀 수 계산
def calculate_masked_pixels(image_path):
    image = cv2.imread(image_path)
    binary_mask = np.where(np.all(image == [0, 0, 0], axis=-1), 0, 1)
    return np.sum(binary_mask)

# 질량 계산
def calculate_mass(area_cm2, thickness_cm, density_g_per_cm3=0.9):
    return area_cm2 * thickness_cm * density_g_per_cm3

# 지방 비율 계산
def calculate_fat_percentage(fat_mask_pixels, beef_mask_pixels):
    return (fat_mask_pixels / beef_mask_pixels) * 100

# 단백질 질량 계산
def calculate_protein_mass(beef_mass, fat_mass, water_percentage=0.725, mineral_percentage=0.01, vitamin_percentage=0.005):
    water_mass = beef_mass * water_percentage
    mineral_mass = beef_mass * mineral_percentage
    vitamin_mass = beef_mass * vitamin_percentage
    return beef_mass - (fat_mass + water_mass + mineral_mass + vitamin_mass)

# 마스크 픽셀 계산
beef_mask_pixels = calculate_masked_pixels('./testpng/maskedtest.png')
fat_mask_pixels = calculate_masked_pixels('./testpng/fat_test.jpeg')

# 면적 및 질량 계산
beef_area_cm2 = beef_mask_pixels / pixels_per_cm2
fat_area_cm2 = fat_mask_pixels / pixels_per_cm2
beef_mass = calculate_mass(beef_area_cm2, meat_thickness_cm)
fat_mass = calculate_mass(fat_area_cm2, meat_thickness_cm) * 0.4

# 비율 및 단백질 질량 계산
fat_percentage = calculate_fat_percentage(fat_mask_pixels, beef_mask_pixels)
protein_mass = calculate_protein_mass(beef_mass, fat_mass)

# 결과 출력
print(f"고기 면적: {beef_area_cm2:.2f}cm²")
print(f"전체 고기 질량: {beef_mass:.2f}g")
print(f"지방 면적: {fat_area_cm2:.2f}cm²")
print(f"지방 질량: {fat_mass:.2f}g")
print(f"지방 비율: {fat_percentage:.2f}%")
print(f"단백질 질량: {protein_mass:.2f}g")
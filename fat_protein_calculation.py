import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands 초기화
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 이미지 경로 설정
image_path = './testpng/test.png'  # 처리할 이미지 파일의 경로를 입력하세요.

# 이미지 읽기
image = cv2.imread(image_path)
if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인해주세요.")
    exit()

# BGR 이미지를 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# MediaPipe Hands 처리
with mp_hands.Hands(
    static_image_mode=True,          # 정적 이미지 처리에 적합한 설정
    max_num_hands=2,                 # 검출할 손의 최대 개수
    min_detection_confidence=0.5     # 최소 검출 신뢰도
) as hands:
    # 손 검출 수행
    results = hands.process(image_rgb)

    # 손이 검출되었는지 확인
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 이미지에 키포인트와 연결선 그리기
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 이미지의 크기
            h, w, c = image.shape

            # 손바닥 길이 계산
            # 손목(0번 키포인트)과 중지 끝(12번 키포인트)의 거리
            x1, y1 = hand_landmarks.landmark[0].x * w, hand_landmarks.landmark[0].y * h
            x2, y2 = hand_landmarks.landmark[12].x * w, hand_landmarks.landmark[12].y * h
            palm_length_pixels = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
            print(f"손바닥 길이 (픽셀): {palm_length_pixels:.2f}")
    else:
        print("이미지에서 손을 검출하지 못했습니다.")
        exit()

# 결과 이미지 출력
cv2.imshow('MediaPipe Hands', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 실제 손바닥 길이 입력 (cm 단위)
palm_length_cm = float(input("실제 손바닥 길이(cm)를 입력하세요: "))

def calculate_pixels_per_cm2(palm_length_pixels, palm_length_cm):
    """
    손바닥의 길이를 이용하여 1cm²당 픽셀 수를 계산합니다.
    
    Parameters:
        palm_length_pixels (float): 이미지에서 측정된 손바닥 길이 (픽셀)
        palm_length_cm (float): 실제 손바닥 길이 (cm)
        
    Returns:
        float: 1cm²당 픽셀 수
    """
    pixels_per_cm = palm_length_pixels / palm_length_cm
    pixels_per_cm2 = pixels_per_cm ** 2
    return pixels_per_cm2

# 1cm²당 픽셀 수 계산
pixels_per_cm2 = calculate_pixels_per_cm2(palm_length_pixels, palm_length_cm)
print(f'1cm²당 픽셀 수: {pixels_per_cm2:.2f}')

def calculate_masked_pixels(image_path):
    """
    이미지에서 검은색(0, 0, 0)을 제외한 모든 픽셀을 1로 변환하고,
    해당하는 픽셀 수를 계산하는 함수입니다.
    
    Parameters:
        image_path (str): 이미지 파일 경로
        
    Returns:
        int: 검은색이 아닌 부분의 픽셀 수
    """
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    # 검은색과 일치하는 픽셀은 0, 나머지는 1로 변환
    binary_mask = np.where(np.all(image == [0, 0, 0], axis=-1), 0, 1)
    
    # 검은색이 아닌 부분의 픽셀 수 계산
    non_black_pixels = np.sum(binary_mask)
    
    return non_black_pixels

def calculate_fat_percentage(fat_mask_pixels, beef_mask_pixels):
    """
    지방의 비율을 계산합니다.
    """
    fat_percentage = (fat_mask_pixels / beef_mask_pixels) * 100
    return fat_percentage

def calculate_mass(area_cm2, thickness_cm=3.81, density_g_per_cm3=0.9):
    """
    질량을 계산합니다.
    면적, 두께, 밀도를 곱하여 계산.
    """
    mass = area_cm2 * thickness_cm * density_g_per_cm3
    return mass

def calculate_protein_mass(beef_mass, fat_mass, water_percentage=0.725, mineral_percentage=0.01, vitamin_percentage=0.005):
    """
    고기의 전체 질량에서 지방, 수분, 무기질, 비타민 및 기타 성분을 뺀 단백질 질량을 계산합니다.
    
    Parameters:
        beef_mass (float): 고기의 전체 질량 (g)
        fat_mass (float): 지방 질량 (g)
        water_percentage (float): 수분 비율 (기본값 72.5%)
        mineral_percentage (float): 무기질 비율 (기본값 1%)
        vitamin_percentage (float): 비타민 및 기타 성분 비율 (기본값 0.5%)
        
    Returns:
        float: 단백질 질량 (g)
    """
    # 수분, 무기질, 비타민 및 기타 성분의 질량을 계산
    water_mass = beef_mass * water_percentage
    mineral_mass = beef_mass * mineral_percentage
    vitamin_mass = beef_mass * vitamin_percentage

    # 단백질 질량 계산
    protein_mass = beef_mass - (fat_mass + water_mass + mineral_mass + vitamin_mass)
    
    return protein_mass

# 3. 지방 마스크 픽셀 수와 소고기 마스크 픽셀 수 계산
image_path_first = './testpng/maskedtest.png'  # 마스킹된 소고기 이미지 파일 경로
image_path_second = './testpng/fat_test.png'  # 마스킹된 지방 이미지 파일 경로

masked_pixels_first = calculate_masked_pixels(image_path_first)
masked_pixels_second = calculate_masked_pixels(image_path_second)
print(f'첫번째 U-net 마스킹된 픽셀 수: {masked_pixels_first}')
print(f'두번째 U-net 마스킹된 픽셀 수: {masked_pixels_second}')

# 4. 지방 비율 계산
fat_percentage = calculate_fat_percentage(masked_pixels_second, masked_pixels_first)
print(f'지방 비율: {fat_percentage:.2f}%')

# 5. 전체 고기 질량 계산
beef_area_cm2 = masked_pixels_first / pixels_per_cm2
beef_mass = calculate_mass(beef_area_cm2)
print(f'전체 고기 질량: {beef_mass:.2f}g')

# 6. 지방 면적 및 지방 질량 계산
fat_area_cm2 = masked_pixels_second / pixels_per_cm2
fat_mass = calculate_mass(fat_area_cm2)
print(f'지방 면적: {fat_area_cm2:.2f}cm²')
print(f'지방 질량: {fat_mass:.2f}g')

# 7. 수분, 무기질, 비타민 및 기타 성분 질량 계산
water_mass = beef_mass * 0.725
mineral_mass = beef_mass * 0.01
vitamin_mass = beef_mass * 0.005
print(f'수분 질량: {water_mass:.2f}g')
print(f'무기질 질량: {mineral_mass:.2f}g')
print(f'비타민 및 기타 성분 질량: {vitamin_mass:.2f}g')

# 8. 단백질 질량 계산
protein_mass = calculate_protein_mass(beef_mass, fat_mass)
print(f'단백질 질량: {protein_mass:.2f}g')
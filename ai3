import cv2
import mediapipe as mp
import time
import numpy as np
import requests

# ESP8266의 IP 주소와 포트 설정
ESP8266_IP = '192.168.150.7'  # ESP8266의 IP 주소 //김지후 핫스팟 기준
ESP8266_PORT = '80'         # ESP8266의 포트 번호

# Mediapipe 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 카메라 사용
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 찾을 수 없습니다.")
    exit()

# 데이터 전송 관련 변수 초기화
last_direction = None  # 마지막으로 전송된 방향
last_sent_time = 0
send_interval = 0.1  # 100ms 간격으로 데이터 전송

# 얼굴 인식 주기 조절 변수
frame_skip_count = 2  # 매 2번째 프레임만 처리
frame_counter = 0

# Mediapipe 얼굴 메쉬 설정
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # 감지할 최대 얼굴 수
    refine_landmarks=True,  # 세부 랜드마크 사용
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    eye_blink_start_time = None  # 눈을 감은 시점
    eye_blink_threshold = 0.007  # 눈이 감겼는지 판단할 임계값
    blink_duration = 0.5  # 눈이 감긴 상태를 감지할 지속 시간 (초)

    while cap.isOpened():
        frame_counter += 1
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            break

        if frame_counter % frame_skip_count != 0:
            continue  # 프레임 스킵

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        direction = "none"  # 초기화
        yaw = None  # yaw 초기화

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
                    # 주요 랜드마크 추출
                    left_eye_outer = face_landmarks.landmark[33]  # 왼쪽 눈 바깥
                    right_eye_outer = face_landmarks.landmark[263]  # 오른쪽 눈 바깥
                    nose_tip = face_landmarks.landmark[1]  # 코 끝
                    left_eye_upper = face_landmarks.landmark[159]  # 왼쪽 눈 위
                    left_eye_lower = face_landmarks.landmark[145]  # 왼쪽 눈 아래

                    # 3D 좌표 변환
                    h, w, _ = image.shape
                    left_eye = np.array([left_eye_outer.x * w, left_eye_outer.y * h, left_eye_outer.z * w])
                    right_eye = np.array([right_eye_outer.x * w, right_eye_outer.y * h, right_eye_outer.z * w])
                    nose = np.array([nose_tip.x * w, nose_tip.y * h, nose_tip.z * w])

                    left_eye_coord = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
                    right_eye_coord = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))
                    nose_tip_coord = (int(nose_tip.x * w), int(nose_tip.y * h))

                    # 랜드마크 표시
                    cv2.circle(image, left_eye_coord, 3, (0, 255, 0), -1)  # 초록색 원
                    cv2.circle(image, right_eye_coord, 3, (255, 0, 0), -1)  # 파란색 원
                    cv2.circle(image, nose_tip_coord, 3, (0, 0, 255), -1)  # 빨간색 원

                    # 얼굴 중심 계산 (두 눈의 중간 지점)
                    face_center = (left_eye + right_eye) / 2
                    center_x = int((left_eye_outer.x + right_eye_outer.x) / 2 * w)

                    # 좌표 기준으로 얼굴의 회전 벡터 계산
                    face_vector = nose - face_center
                    face_vector /= np.linalg.norm(face_vector)  # 정규화

                    # 얼굴 회전 각도 계산
                    yaw = np.arctan2(face_vector[0], face_vector[2]) * 180 / np.pi

                    # 왼쪽 눈 감김 계산
                    left_eye_dist = abs(left_eye_upper.y - left_eye_lower.y)

                    # 눈 깜빡임 감지
                    if left_eye_dist < eye_blink_threshold:
                        if eye_blink_start_time is None:
                            eye_blink_start_time = time.time()
                        elif time.time() - eye_blink_start_time >= blink_duration:
                            direction = "stop"
                    else:
                        eye_blink_start_time = None

                    # 얼굴 방향 결정 (Yaw를 기준으로 설정)
                    if direction not in ["stop", "unknown"]:
                        if yaw <= -160 or yaw >= 160:
                            direction = "front"
                        elif yaw < 0:
                            if yaw > -110:
                                direction = "left++"
                            elif yaw > -140:
                                direction = "left+"
                            elif yaw > -160:
                                direction = "left"
                        else:
                            if yaw < 110:
                                direction = "right++"
                            elif yaw < 140:
                                direction = "right+"
                            elif yaw < 160:
                                direction = "right"

                except (IndexError, AttributeError):
                    direction = "unknown"

        else:
            direction = "unknown"

        # 방향 및 각도 텍스트 표시
        if yaw is not None:
            cv2.putText(image, f'{direction}  Yaw: {int(yaw)}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f'{direction}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # 화면 출력
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

        # ESP8266으로 명령 전송
        if direction != "unknown" and direction != last_direction:
            current_time = time.time()
            if current_time - last_sent_time > send_interval:
                try:
                    url = f"http://{ESP8266_IP}:{ESP8266_PORT}/{direction}"
                    response = requests.get(url)
                    print(f"Sent direction {direction} to ESP8266: {response.status_code}")
                    last_direction = direction  # 마지막 전송 방향 갱신
                    last_sent_time = current_time  # 마지막 전송 시간 갱신
                except Exception as e:
                    print(f"Failed to send direction to ESP8266: {e}")

cap.release()
cv2.destroyAllWindows()

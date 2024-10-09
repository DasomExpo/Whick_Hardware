import cv2  # OpenCV 라이브러리를 임포트하여 영상 처리 기능 사용
import mediapipe as mp  # Mediapipe 라이브러리를 임포트하여 얼굴 인식 등 사용
import time  # 시간 관련 함수 사용을 위한 라이브러리 임포트
import numpy as np  # 수치 계산을 위한 NumPy 라이브러리 임포트
import requests  # HTTP 요청을 보내기 위한 requests 라이브러리 임포트

# ESP8266의 IP 주소와 포트 설정
ESP8266_IP = '192.168.83.7'  # ESP8266 모듈의 IP 주소
ESP8266_PORT = '80'           # ESP8266 모듈의 포트 번호

# Mediapipe 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh  # 얼굴 메쉬 기능을 사용하기 위한 초기화
mp_drawing = mp.solutions.drawing_utils  # 랜드마크 그리기 등의 유틸리티 함수 사용

# 카메라 사용 설정
cap = cv2.VideoCapture(0)  # 기본 카메라 장치를 열어서 영상 캡처 시작
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # 카메라 프레임 너비를 320으로 설정하여 해상도 줄이기
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 카메라 프레임 높이를 240으로 설정

# 카메라가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 찾을 수 없습니다.")  # 오류 메시지 출력
    exit()  # 프로그램 종료

# 데이터 전송 관련 변수 초기화
last_direction = None  # 마지막으로 전송된 방향을 저장할 변수
last_sent_time = 0  # 마지막으로 데이터를 전송한 시간을 저장할 변수
send_interval = 1.0  # 데이터를 전송하는 간격을 1초로 설정

# 얼굴 인식 주기 조절 변수
frame_skip_count = 20  # 매 20번째 프레임마다 얼굴 인식 수행
frame_counter = 0  # 프레임 카운터 초기화

# Mediapipe 얼굴 메쉬 설정
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # 감지할 최대 얼굴 수를 1로 설정
    refine_landmarks=True,  # 더 자세한 랜드마크 사용
    min_detection_confidence=0.5,  # 얼굴 감지 최소 신뢰도
    min_tracking_confidence=0.5  # 얼굴 추적 최소 신뢰도
) as face_mesh:

    eye_blink_start_time = None  # 눈을 감은 시점을 저장할 변수
    eye_blink_threshold = 0.007  # 눈이 감겼는지 판단하는 임계값
    blink_duration = 0.5  # 눈이 감긴 상태를 감지할 지속 시간 (초)
    toggle_stop = False  # "stop" 상태를 토글하기 위한 플래그 변수
    head_up = False  # 고개를 들었는지 여부를 확인하는 플래그 변수

    while cap.isOpened():  # 카메라가 열려 있는 동안 반복
        frame_counter += 1  # 프레임 카운터 증가
        success, image = cap.read()  # 카메라에서 프레임 읽기
        if not success:
            print("카메라를 찾을 수 없습니다.")  # 프레임을 읽지 못하면 오류 메시지 출력
            break  # 반복문 종료

        if frame_counter % frame_skip_count != 0:
            continue  # 지정된 프레임 간격이 아니면 다음 반복으로 넘어감 (프레임 스킵)

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # 이미지 좌우 반전 및 색상 공간 변환
        image.flags.writeable = False  # 이미지에 쓰기 방지하여 성능 향상
        results = face_mesh.process(image)  # 얼굴 메쉬 처리

        image.flags.writeable = True  # 이미지에 다시 쓰기 허용
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지 색상 공간을 다시 BGR로 변환

        direction = "none"  # 방향 변수 초기화
        yaw = None  # 좌우 회전 각도 (Yaw) 초기화
        pitch = None  # 상하 회전 각도 (Pitch) 초기화

        if results.multi_face_landmarks:  # 얼굴 랜드마크가 감지되었을 때
            for face_landmarks in results.multi_face_landmarks:
                try:
                    # 주요 랜드마크 추출
                    left_eye_outer = face_landmarks.landmark[33]  # 왼쪽 눈 바깥쪽 랜드마크
                    right_eye_outer = face_landmarks.landmark[263]  # 오른쪽 눈 바깥쪽 랜드마크
                    nose_tip = face_landmarks.landmark[1]  # 코 끝 랜드마크
                    left_eye_upper = face_landmarks.landmark[159]  # 왼쪽 눈 위쪽 랜드마크
                    left_eye_lower = face_landmarks.landmark[145]  # 왼쪽 눈 아래쪽 랜드마크
                        
                    # 3D 좌표 변환
                    h, w, _ = image.shape  # 이미지의 높이(h), 너비(w) 가져오기
                    left_eye = np.array([left_eye_outer.x * w, left_eye_outer.y * h, left_eye_outer.z * w])  # 왼쪽 눈 좌표 계산
                    right_eye = np.array([right_eye_outer.x * w, right_eye_outer.y * h, right_eye_outer.z * w])  # 오른쪽 눈 좌표 계산
                    nose = np.array([nose_tip.x * w, nose_tip.y * h, nose_tip.z * w])  # 코 끝 좌표 계산

                    left_eye_coord = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))  # 왼쪽 눈 좌표 (정수형)
                    right_eye_coord = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))  # 오른쪽 눈 좌표
                    nose_tip_coord = (int(nose_tip.x * w), int(nose_tip.y * h))  # 코 끝 좌표

                    # 랜드마크 표시 (이미지에 원 그리기)
                    cv2.circle(image, left_eye_coord, 3, (0, 255, 0), -1)  # 왼쪽 눈에 초록색 원 그리기
                    cv2.circle(image, right_eye_coord, 3, (255, 0, 0), -1)  # 오른쪽 눈에 파란색 원 그리기
                    cv2.circle(image, nose_tip_coord, 3, (0, 0, 255), -1)  # 코 끝에 빨간색 원 그리기
                        
                    # 얼굴 중심 계산 (두 눈의 중간 지점)
                    face_center = (left_eye + right_eye) / 2  # 얼굴 중심 좌표 계산
                    face_vector = nose - face_center  # 얼굴 벡터 계산 (코 끝과 얼굴 중심의 차이)
                    face_vector /= np.linalg.norm(face_vector)  # 벡터 정규화 (크기를 1로 만듦)

                    # 얼굴 회전 각도 계산
                    yaw = np.arctan2(face_vector[0], face_vector[2]) * 180 / np.pi  # 좌우 회전 각도 계산
                    pitch = np.arctan2(face_vector[1], face_vector[2]) * 180 / np.pi  # 상하 회전 각도 계산

                    # 왼쪽 눈 감김 정도 계산
                    left_eye_dist = abs(left_eye_upper.y - left_eye_lower.y)  # 왼쪽 눈 위아래 랜드마크의 y좌표 차이
                        
                    # 눈 깜빡임 감지
                    if left_eye_dist < eye_blink_threshold:  # 눈이 감겼을 때
                        if eye_blink_start_time is None:
                            eye_blink_start_time = time.time()  # 눈을 감은 시간 기록
                        elif time.time() - eye_blink_start_time >= blink_duration:
                            direction = "stop"  # 일정 시간 이상 눈을 감았으면 "stop"으로 설정
                    else:
                        eye_blink_start_time = None  # 눈을 뜬 상태이면 시간 초기화

                    # 고개를 들었을 때
                    if (pitch < -180 or pitch > 165) and not head_up:
                        head_up = True  # 고개를 든 상태로 변경
                        toggle_stop = not toggle_stop  # "stop" 상태를 토글

                    # 고개를 내렸을 때
                    elif (-180 <= pitch <= 160) and head_up:
                        head_up = False  # 고개를 내린 상태로 변경

                    # "stop" 상태일 때 방향을 "stop"으로 설정
                    if toggle_stop:
                        direction = "stop"                            

                    # 얼굴 방향 결정 (Yaw 각도를 기준으로)
                    if not toggle_stop:  # "stop" 상태가 아닐 때만 방향 계산
                        if direction not in ["stop", "unknown"]:
                            if yaw <= -160 or yaw >= 160:
                                direction = "front"  # 정면
                            elif yaw < 0:
                                if yaw > -132:
                                    direction = "left++"  # 왼쪽 약간
                                elif yaw > -145:
                                    direction = "left+"  # 왼쪽 조금 더
                                elif yaw > -160:
                                    direction = "left"  # 왼쪽 많이
                            else:
                                if yaw < 132:
                                    direction = "right++"  # 오른쪽 약간
                                elif yaw < 145:
                                    direction = "right+"  # 오른쪽 조금 더
                                elif yaw < 160:
                                    direction = "right"  # 오른쪽 많이     
                    
                except (IndexError, AttributeError):
                    direction = "unknown"  # 랜드마크 인덱스 오류나 속성 오류 발생 시 방향을 "unknown"으로 설정

        else:
            direction = "unknown"  # 얼굴 랜드마크를 찾지 못했을 때

        # 방향 및 각도 텍스트를 이미지에 표시
        if yaw is not None and pitch is not None:
            cv2.putText(image, f'{direction}  Yaw: {int(yaw)}, Pitch: {int(pitch)}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f'{direction}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # ESP8266으로 명령 전송
        if direction != "unknown" and direction != last_direction:  # 방향이 변경되었을 때
            current_time = time.time()
            if current_time - last_sent_time > send_interval:  # 전송 간격이 지났을 때
                try:
                    url = f"http://{ESP8266_IP}:{ESP8266_PORT}/{direction}"  # 요청할 URL 생성
                    response = requests.get(url, timeout=1)  # HTTP GET 요청 (타임아웃 1초)
                    print(f"Sent direction {direction} to ESP8266: {response.status_code}")  # 전송 결과 출력
                    last_direction = direction  # 마지막 전송된 방향 갱신
                    last_sent_time = current_time  # 마지막 전송 시간 갱신
                except requests.exceptions.Timeout:
                    print(f"Timeout when sending direction {direction} to ESP8266")  # 타임아웃 예외 처리
                except requests.exceptions.ConnectionError:
                    print(f"Connection error when sending direction {direction} to ESP8266")  # 연결 오류 예외 처리
                except Exception as e:
                    print(f"An error occurred: {e}")  # 기타 예외 처리

        # 화면에 이미지 출력
        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

    cap.release()  # 카메라 장치 해제
    cv2.destroyAllWindows()  # 모든 창 닫기


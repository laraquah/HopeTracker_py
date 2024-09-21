import cv2
import numpy as np
import os
from openvino.runtime import Core
from sklearn import neighbors
import time

# 변수 초기화
data_dir = './collected_data'
model_path = "Models\\face-detection-adas-0001.xml"
images = []
labels = []
trained = False
clf = None

# Inference Engine 초기화
core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)

# 웹캠 시작
cap = cv2.VideoCapture(0)

# 데이터 수집 함수
def collect_data(frame):
    filename = os.path.join(data_dir, f'image_{len(images)}.png')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cv2.imwrite(filename, frame)
    images.append(filename)
    labels.append(1)
    print(f"Image saved: {filename}")

    # 데이터 수집 중임을 화면에 표시
    cv2.putText(frame, "Collecting data...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)
    cv2.waitKey(1000)  # 1초 동안 표시

# 모델 학습 함수
def train_model():
    global clf, trained
    print("Training model...")
    X = []
    y = []
    for img_path, label in zip(images, labels):
        img = cv2.imread(img_path)
        if img is not None:  # 이미지가 유효한 경우에만 처리
            img_resized = cv2.resize(img, (128, 128)).flatten()
            X.append(img_resized)
            y.append(label)
        else:
            print(f"Warning: Unable to read image {img_path}. Skipping...")
    
    if not X:  # 유효한 이미지가 없는 경우
        print("Error: No valid images for training.")
        return

    clf = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf.fit(X, y)
    trained = True
    print("Model trained successfully!")

    # 학습 완료 표시
    start_time = time.time()
    while time.time() - start_time < 5:
        display_message("Model Training Complete", (0, 255, 0))

# 얼굴 탐지 함수
def detect_faces():
    detection_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (672, 384))
        frame_input = frame_resized.transpose((2, 0, 1)).reshape(1, 3, 384, 672)
        result = compiled_model([frame_input])
        missing_person_detected = False
        
        for obj in result[compiled_model.output(0)][0][0]:
            if obj[2] > 0.5:
                xmin = int(obj[3] * frame.shape[1])
                ymin = int(obj[4] * frame.shape[0])
                xmax = int(obj[5] * frame.shape[1])
                ymax = int(obj[6] * frame.shape[0])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                face = frame[ymin:ymax, xmin:xmax]
                face_resized = cv2.resize(face, (128, 128)).flatten().reshape(1, -1)
                if trained and clf.predict(face_resized)[0] == 1:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    missing_person_detected = True
                    detection_time = time.time()
                    display_message("Missing person detected!", (0, 255, 0))  # 초록색 문구 표시

        if not missing_person_detected:
            display_message("No missing person detected", (0, 0, 255))  # 빨간색 문구 표시

        cv2.imshow("Face Detection", frame)

        # 실종자 감지 후 10초 지나면 프로그램 종료
        if detection_time and time.time() - detection_time > 10:
            print("Program will close after 10 seconds since detection.")
            break

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 화면에 메시지를 표시하는 함수
def display_message(text, color):
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Message", frame)
    cv2.waitKey(1)

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam", frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('d'):
        collect_data(frame)
    elif key == ord('o'):
        train_model()
    elif key == ord('p'):
        detect_faces()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

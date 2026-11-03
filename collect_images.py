import cv2
import os

SAVE_DIR = "dataset_raw"

os.makedirs(SAVE_DIR, exist_ok=True)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = len([
    f for f in os.listdir(SAVE_DIR)
    if f.lower().endswith((".jpg", ".png"))
])

print("웹캠 데이터 수집 시작")
print("S : 사진 저장")
print("Q : 종료")

while True:
    success, frame = camera.read()

    if not success:
        print("카메라를 읽을 수 없습니다.")
        break

    display_frame = frame.copy()

    cv2.putText(
        display_frame,
        f"Saved: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        "S: Save / Q: Quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Collector", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # S를 누르면 이미지 저장
    if key == ord("s"):
        filename = os.path.join(
            SAVE_DIR,
            f"img_{count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

        count += 1

    # Q를 누르면 종료
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("총 저장 이미지:", count)
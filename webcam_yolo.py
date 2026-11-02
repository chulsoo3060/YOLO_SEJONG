import time

import cv2
from ultralytics import YOLO


def main():
    # 가벼운 사전 학습 모델 사용: 최초 실행 시 자동 다운로드
    model = YOLO("yolo11n.pt")

    # Windows의 기본 카메라 열기
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    try:
        if not camera.isOpened():
            raise RuntimeError(
                "카메라를 열 수 없습니다. 카메라 권한과 다른 앱의 사용 여부를 확인하세요."
            )

        # 카메라에 해상도 요청: 실제 지원 해상도는 장치에 따라 달라짐
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("실시간 인식 시작: 영상 창에서 Q 또는 ESC를 누르면 종료합니다.")

        while True:
            start_time = time.perf_counter()

            # 1. 카메라에서 이미지 한 장 읽기
            success, frame = camera.read()
            if not success:
                raise RuntimeError("카메라에서 영상을 읽지 못했습니다.")

            # 2. 프레임에 있는 객체 감지
            result = model.predict(
                source=frame,
                device="cpu",
                imgsz=416,
                conf=0.4,
                verbose=False,
            )[0]

            # 3. 감지된 객체의 박스, 이름, 신뢰도 그리기
            output_frame = result.plot()

            # 카메라 읽기부터 결과 그리기까지의 처리 속도
            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                output_frame,
                f"Processing FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # 4. 결과 화면 표시
            cv2.imshow("YOLO Webcam", output_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            # 창의 X 버튼으로도 종료
            if cv2.getWindowProperty("YOLO Webcam", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        # 오류나 종료 시에도 카메라와 창 정리
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
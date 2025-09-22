import cv2 as cv

def list_cameras(max_test=5):
    # Try to open the first N camera indices and print which are available.
    print("Checking available cameras...")
    for i in range(max_test):
        cap = cv.VideoCapture(i, cv.CAP_AVFOUNDATION)  # force AVFoundation backend
        if cap.isOpened():
            print(f"Camera {i} is available")
            cap.release()

def open_camera(index=0, width=1280, height=720):
    # Open a specific camera index with given resolution.
    cap = cv.VideoCapture(index, cv.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")

    # Try to set resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    print(f"Opened camera {index} at {width}x{height}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed")
            break

        cv.imshow(f"Camera {index}", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):  # press 'q' to quit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    list_cameras(max_test=3)
    open_camera(index=0)

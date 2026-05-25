"""
Face Detection System using OpenCV
Detects human faces in images, video files, or live webcam feed.
"""

import cv2
import sys
import os


def load_detector():
    """Load the Haar Cascade face detector."""
    # OpenCV ships with this XML file
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade from: {cascade_path}")
    return detector


def detect_faces(frame, detector, scale=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Detect faces in a single frame.

    Args:
        frame        : BGR image (numpy array)
        detector     : cv2.CascadeClassifier
        scale        : scaleFactor — how much the image is reduced at each scale
        min_neighbors: higher = fewer but stronger detections
        min_size     : minimum face size in pixels

    Returns:
        List of (x, y, w, h) rectangles for each detected face
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)           # improve contrast
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces if len(faces) else []


def draw_faces(frame, faces):
    """Draw bounding boxes and labels on detected faces."""
    for i, (x, y, w, h) in enumerate(faces, start=1):
        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Label background
        label = f"Face {i}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 4, y), (0, 255, 0), -1)
        # Label text
        cv2.putText(
            frame, label,
            (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 2,
        )
    # Face count overlay
    cv2.putText(
        frame, f"Faces detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 200, 255), 2,
    )
    return frame


# ──────────────────────────────────────────────
# Mode 1: Detect in a static image
# ──────────────────────────────────────────────
def detect_image(image_path: str, output_path: str = "output.jpg"):
    detector = load_detector()
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    faces = detect_faces(frame, detector)
    print(f"[INFO] Detected {len(faces)} face(s) in '{image_path}'")

    result = draw_faces(frame.copy(), faces)
    cv2.imwrite(output_path, result)
    print(f"[INFO] Result saved to '{output_path}'")

    cv2.imshow("Face Detection – press any key to close", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Mode 2: Detect in a video file or webcam
# ──────────────────────────────────────────────
def detect_video(source=0, output_path: str = None):
    """
    Args:
        source      : 0 for webcam, or a path string for a video file
        output_path : optional path to save the annotated video
    """
    detector = load_detector()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame, detector)
        annotated = draw_faces(frame, faces)

        cv2.imshow("Face Detection – press q to quit", annotated)
        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Video saved to '{output_path}'")
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Usage:
        python face_detection.py                        # live webcam
        python face_detection.py photo.jpg              # static image
        python face_detection.py video.mp4              # video file
        python face_detection.py photo.jpg out.jpg      # image + save result
        python face_detection.py video.mp4 out.mp4      # video + save result
    """
    args = sys.argv[1:]

    if not args:
        # No arguments → live webcam
        detect_video(source=0)

    elif len(args) == 1:
        src = args[0]
        ext = os.path.splitext(src)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            detect_image(src)
        else:
            detect_video(source=src)

    elif len(args) == 2:
        src, dst = args
        ext = os.path.splitext(src)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            detect_image(src, output_path=dst)
        else:
            detect_video(source=src, output_path=dst)

    else:
        print("Usage: python face_detection.py [source] [output]")
        sys.exit(1)
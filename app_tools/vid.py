import cv2
from PIL import Image
import os
import uuid
from typing import List, Dict

def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 0.5,
    max_frames: int | None = 20
) -> List[Dict[str, str]]:
    """
    Extract frames from a video at a given rate and save them as JPGs.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        fps (float): Frames per second to extract (0.5 = one frame every 2 seconds).
        max_frames (int | None): Maximum number of frames to save (None = all).

    Returns:
        List[Dict[str, str]]: List of dicts with frame metadata:
            [
                {"frame_id": "uuid-0000", "path": "output/frame.jpg"},
                ...
            ]
    """
    os.makedirs(output_dir, exist_ok=True)

    # Universal UUID for this video run
    video_uuid = uuid.uuid4().hex
    print(f" Universal video UUID: {video_uuid}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f" Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps) if fps > 0 else 1

    frame_count = 0
    saved_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_id = f"{video_uuid}-{saved_count:04d}"
            save_path = os.path.join(output_dir, f"{frame_id}.jpg")

            img.save(save_path)
            print(f" Saved {save_path}")

            results.append({"frame_id": frame_id, "path": save_path})
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(" Done extracting frames!")
    return results

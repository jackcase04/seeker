"""
Usage:
    python extract_poses.py video/mp4/TestVideo.mp4 data/workspace/pose/test/
"""

import sys
import json
import os
import cv2
from ultralytics import YOLO

def extract(video_path, out_dir):
    model_name = "yolov8m-pose.pt"

    stem = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(out_dir, f"{stem}_alphapose_tracked_person.json")
    vis_path = os.path.join(out_dir, f"{stem}_annotated.mp4")

    model = YOLO(model_name)

    # Set up OpenCV with all the information from the captured video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up a writer to write the annotated video
    writer = cv2.VideoWriter(vis_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pose_dict = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(f"Processing frame {frame_idx}/{total}")

        results = model.track(frame, persist=True, verbose=False)
        result = results[0]

        frame_entry = {}

        if result.keypoints is not None and result.boxes is not None:
            kps_xy = result.keypoints.xy.cpu().numpy()    # (N, 17, 2)
            kps_conf = result.keypoints.conf.cpu().numpy() # (N, 17)
            boxes = result.boxes

            ids = boxes.id
            confs = boxes.conf.cpu().numpy()

            for person_idx in range(len(kps_xy)):
                track_id = int(ids[person_idx].item()) if ids is not None else person_idx
                person_id = f"{track_id:04d}"

                xy = kps_xy[person_idx]    # (17, 2)
                cf = kps_conf[person_idx]  # (17,)

                keypoints = []
                for joint in range(17):
                    keypoints.extend([float(xy[joint, 0]), float(xy[joint, 1]), float(cf[joint])])

                frame_entry[person_id] = {
                    "keypoints": keypoints,
                    "scores": float(confs[person_idx]),
                }

        pose_dict[str(frame_idx)] = frame_entry

        annotated = result.plot()
        writer.write(annotated)

    cap.release()
    writer.release()
    print()

    with open(json_path, "w") as f:
        json.dump(pose_dict, f)

extract(sys.argv[1], sys.argv[2])
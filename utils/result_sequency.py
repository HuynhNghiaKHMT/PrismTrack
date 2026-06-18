import cv2
import os
import numpy as np
import configparser

def render_tracker_video(img_path, txt_path, video_path, frame_path, fps=30, is_score=1):

    results = {}

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 7:
                continue

            try:
                f_id = int(parts[0])
                obj_id = int(parts[1])

                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                conf_val = float(parts[6])

            except ValueError:
                continue

            if f_id not in results:
                results[f_id] = []

            results[f_id].append(
                [obj_id, x, y, w, h, conf_val]
            )

    images = sorted(
        [
            img
            for img in os.listdir(img_path)
            if img.endswith((".jpg", ".png"))
        ]
    )

    first_img = cv2.imread(
        os.path.join(img_path, images[0])
    )

    height, width, _ = first_img.shape

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    os.makedirs(frame_path, exist_ok=True)

    np.random.seed(42)
    colors = np.random.randint(
        0,
        255,
        size=(1000, 3),
        dtype=np.uint8,
    )

    for i, img_name in enumerate(images):

        frame_id = i + 1

        frame = cv2.imread(
            os.path.join(img_path, img_name)
        )

        if frame_id in results:

            for obj in results[frame_id]:

                oid, x, y, w, h, s = obj

                color = colors[oid % 1000].tolist()

                cv2.rectangle(
                    frame,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    color,
                    2,
                )

                if is_score:
                    label_text = (f"{oid}: {s:.2f}")
                else:
                    label_text = f"{oid}"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                (lw, lh), _ = cv2.getTextSize(
                    label_text,
                    font,
                    font_scale,
                    thickness,
                )

                cv2.rectangle(
                    frame,
                    (int(x), int(y) - lh - 10),
                    (int(x) + lw + 10, int(y)),
                    color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label_text,
                    (int(x) + 5, int(y) - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # Ghi video
        out.write(frame)

        # Lưu frame
        cv2.imwrite(
            os.path.join(
                frame_path,
                f"{frame_id:06d}.jpg",
            ),
            frame,
        )

    out.release()

    print(f"Video path : {video_path}")
    print(f"Frame path : {frame_path}")


if __name__ == "__main__":

    DIR_PATH = r"dataset/DanceTrack/val/dancetrack0065"

    IMG_PATH = os.path.join(DIR_PATH, "img1")
    TXT_PATH = os.path.join(DIR_PATH, "gt", "gt.txt")

    VIDEO_PATH = r"assets/Dance/dancetrack0065.mp4"
    FRAME_PATH = r"assets/Dance/img"

    seqinfo_path = os.path.join(DIR_PATH, "seqinfo.ini")

    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    fps = config.getint("Sequence", "frameRate")

    render_tracker_video(IMG_PATH, TXT_PATH, VIDEO_PATH, FRAME_PATH, fps=fps, is_score=0,)
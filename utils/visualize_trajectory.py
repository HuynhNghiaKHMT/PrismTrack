import os
import matplotlib.pyplot as plt
import numpy as np


def load_trajectory_data(txt_path, target_id, max_frames=100):

    if txt_path is None or not os.path.exists(txt_path):
        return None, None

    xs = []
    ys = []

    with open(txt_path, "r") as f:

        for line in f:

            parts = line.strip().split(",")

            if len(parts) < 6:
                continue

            try:
                frame_id = int(parts[0])
                obj_id = int(parts[1])

                if frame_id > max_frames:
                    continue

                if obj_id != target_id:
                    continue

                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                cx = x + w / 2.0
                cy = y + h / 2.0

                xs.append(cx)
                ys.append(cy)

            except ValueError:
                continue

    return np.asarray(xs), np.asarray(ys)


def plot_multi_trajectories(
    gt_path, prism_track_path, track_track_path,
    gt_id, prism_track_id, track_track_id,
    title_name="DanceTrack", max_frames=100, marker_step=5,
):

    # =====================================================
    # Load trajectories
    # =====================================================

    gt_x, gt_y = load_trajectory_data(gt_path, gt_id, max_frames)

    prism_x, prism_y = load_trajectory_data(prism_track_path, prism_track_id, max_frames)

    track_x, track_y = load_trajectory_data(track_track_path, track_track_id, max_frames)

    if gt_x is None or len(gt_x) == 0:
        print(f"Cannot find GT ID={gt_id}")
        return

    # =====================================================
    # Compute plotting range
    # =====================================================

    all_x = [gt_x]
    all_y = [gt_y]

    if prism_x is not None and len(prism_x):
        all_x.append(prism_x)
        all_y.append(prism_y)

    if track_x is not None and len(track_x):
        all_x.append(track_x)
        all_y.append(track_y)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)

    x_range = max(x_range, 1)
    y_range = max(y_range, 1)

    # Figure tự động theo tỷ lệ dữ liệu
    fig_w = 8
    fig_h = fig_w * (y_range / x_range)

    fig_h = np.clip(fig_h, 3, 8)

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h),
        facecolor="white"
    )

    ax.set_facecolor("white")

    # =====================================================
    # Ground Truth
    # =====================================================

    ax.plot(
        gt_x, gt_y,
        color="black",
        linewidth=3.0,
        marker="x",
        markersize=4,
        markevery=marker_step,
        label=f"GT #{gt_id}"
    )

    # =====================================================
    # PRISM_TRACK
    # =====================================================

    if prism_x is not None and len(prism_x):

        ax.plot(
            prism_x,
            prism_y,
            color="red",
            linewidth=1.8,
            label=f"PRISM_TRACK #{prism_track_id}"
        )

        ax.scatter(
            prism_x[::marker_step],
            prism_y[::marker_step],
            color="red",
            s=18,
            zorder=3
        )

    # =====================================================
    # TRACK_TRACK
    # =====================================================

    if track_x is not None and len(track_x):

        ax.plot(
            track_x,
            track_y,
            color="green",
            linewidth=1.8,
            label=f"TRACK_TRACK #{track_track_id}"
        )

        ax.scatter(
            track_x[::marker_step],
            track_y[::marker_step],
            color="green",
            marker="^",
            s=20,
            zorder=4
        )

    # =====================================================
    # IMPORTANT
    # =====================================================

    # ax.set_aspect("equal", adjustable="box")
    ax.set_aspect(6.0, adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")

    # legend
    ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=True
    )

    # caption
    ax.text(
        0.5,
        -0.08,
        f"(a) {title_name}",
        transform=ax.transAxes,
        ha="center",
        fontsize=12
    )

    plt.tight_layout(pad=0.3)

    output_name = (
        f"trajectory_GT{gt_id}"
        f"_PRISM{prism_track_id}"
        f"_TRACK{track_track_id}.png"
    )

    save_dir = "assets"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(save_dir, output_name),
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()


if __name__ == "__main__":

    GT_TXT = r"../dataset/DanceTrack/val/dancetrack0026/gt/gt.txt"
    PRISM_TRACK_TXT = (r"../outputs/3. track/newtrack/dance_val_post/dancetrack0026.txt")
    TRACK_TRACK_TXT = (r"../outputs/3. track/dancetrack_val_post/dancetrack0026.txt")

    plot_multi_trajectories(
        gt_path=GT_TXT, prism_track_path=PRISM_TRACK_TXT, track_track_path=TRACK_TRACK_TXT,
        gt_id=10, prism_track_id=13, track_track_id=10,
        title_name="DanceTrack0026", max_frames=120, marker_step=1
    )
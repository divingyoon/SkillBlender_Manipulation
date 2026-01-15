#/home/user/rl_ws/IsaacLab/_isaac_sim/python.sh /home/user/rl_ws/save_tb_pngs.py
"""Export all TensorBoard scalar tags to PNG plots."""

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Update these paths as needed or set via env vars.
LOG_DIR = os.environ.get("TB_LOG_DIR", "/home/user/rl_ws/IsaacLab/logs/rsl_rl/openarm_bi_grasp_2g/test5")
OUT_DIR = os.environ.get("TB_OUT_DIR", "/home/user/rl_ws/result")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(f"Log dir not found: {LOG_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    acc = event_accumulator.EventAccumulator(
        LOG_DIR, size_guidance={event_accumulator.SCALARS: 0}
    )
    acc.Reload()

    tags = acc.Tags().get("scalars", [])
    if not tags:
        raise SystemExit("No scalar tags found. Check TB_LOG_DIR.")

    for tag in tags:
        events = acc.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure()
        plt.plot(steps, values)
        plt.title(tag)
        plt.xlabel("step")
        plt.ylabel("value")
        safe_name = tag.replace("/", "_")
        out_path = os.path.join(OUT_DIR, f"{safe_name}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Saved {len(tags)} plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()

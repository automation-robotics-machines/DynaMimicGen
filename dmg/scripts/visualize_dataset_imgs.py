import os
import numpy as np
import h5py
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import argparse
import dmg.demos

class ImageBrowser:
    def __init__(self, demo, gripper_actions, agentview_images, eye_in_hand_images):
        self.agentview_images = agentview_images
        self.eye_in_hand_images = eye_in_hand_images
        self.index = 0
        self.demo = demo

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.suptitle(f"{self.demo} | Frame {self.index + 1}/{len(self.agentview_images)} | Gripper Action: {gripper_actions[self.index]}", fontsize=12)
        plt.subplots_adjust(bottom=0.2)

        self.agent_img_display = self.ax1.imshow(self.agentview_images[self.index])
        self.ax1.set_title("Agent View")

        self.eye_img_display = self.ax2.imshow(self.eye_in_hand_images[self.index])
        self.ax2.set_title("Robot Eye-In-Hand View")

        # Add buttons
        axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.1, 0.075])

        self.btn_prev = Button(axprev, 'Previous')
        self.btn_next = Button(axnext, 'Next')

        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)

        plt.show()

    def update_images(self):
        """ Update displayed images """
        self.agent_img_display.set_data(self.agentview_images[self.index])
        self.eye_img_display.set_data(self.eye_in_hand_images[self.index])
        self.fig.suptitle(f"{self.demo} | Frame {self.index + 1}/{len(self.agentview_images)} | Gripper Action: {gripper_actions[self.index]}", fontsize=12)
        self.ax1.set_title(f"Agent View (Frame {self.index + 1}/{len(self.agentview_images)})")
        self.ax2.set_title(f"Robot Eye-In-Hand View (Frame {self.index + 1}/{len(self.eye_in_hand_images)})")
        self.fig.canvas.draw()

    def next_image(self, event):
        """ Display the next image in the sequence """
        if self.index < len(self.agentview_images) - 1:
            self.index += 1
            self.update_images()

    def prev_image(self, event):
        """ Display the previous image in the sequence """
        if self.index > 0:
            self.index -= 1
            self.update_images()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, 
                        help="Path to your demonstration directory that contains the demo.hdf5 file, e.g.: 'path_to_demos_dir/hdf5/YOUR_DEMONSTRATION'")
    parser.add_argument(
        "--play-dmp",
        action="store_true",
    )
    parser.add_argument(
        "--demos", type=str, default="all",
        help="Comma-separated list of demo keys to keep, e.g., 'demo_0,demo_1,demo_5'"
    )
    args = parser.parse_args()

    dir_path = os.path.join(dmg.demos.hdf5_root, args.directory)
    if args.play_dmp:
        dataset_path = os.path.join(dir_path, "dmp/image.hdf5")
    else:
        dataset_path = os.path.join(dir_path, "image.hdf5")

    with h5py.File(dataset_path, "r") as f:
        if args.demos != "all":
            selected_demos = [demo.strip() for demo in args.demos.split(",")]
        else:
            selected_demos = list(f["data"].keys())

        ep = selected_demos[0]
        
        gripper_actions = np.array(f[f"data/{ep}/actions"])[:, -1]
        robot0_eye_in_hand_image = np.array(f[f"data/{ep}/obs/robot0_eye_in_hand_image"])
        agentview_image = np.array(f[f"data/{ep}/obs/agentview_image"])

    # Launch interactive viewer
    ImageBrowser(ep, gripper_actions, agentview_image, robot0_eye_in_hand_image)

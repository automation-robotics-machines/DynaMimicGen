import numpy as np
import matplotlib.pyplot as plt
import copy
from dmg.dmp import JointDMP
from dmg import utils

def rollout_dmp_to_trajectory(dmp, ts, tau):
    """Rollout a trained DMP to generate a new trajectory."""
    # dmp_copy = copy.deepcopy(dmp)
    p, dp, ddp = dmp.rollout(ts=ts, tau=tau, FX=True)
    return p


if __name__ == "__main__":
    # Example usage of the above functions
    np.random.seed(42)
    N = 200
    dof1 = np.array(np.sin(np.linspace(0, np.pi/3, N))).reshape(N, 1)  # Example
    dof2 = np.array(np.cos(np.linspace(0, np.pi/3, N))).reshape(N, 1)  # Example
    traj = np.hstack([dof1, dof2])
    env_name = "ExampleEnv"

    # Train DMP
    dmp, ts, tau = utils.train_dmps([traj], env_name)

    # Rollout DMP
    generated_traj = rollout_dmp_to_trajectory(dmp[0], ts[0], tau[0])

    print("DMP training and rollout completed.")

    # Plot demonstration vs DMP trajectories
    labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    n_dof = traj.shape[1]

    fig, axs = plt.subplots(n_dof, 1, figsize=(8, 12), sharex=True)
    fig.suptitle("Demonstration vs DMP Trajectories", fontsize=14)

    for i in range(n_dof):
        axs[i].plot(generated_traj[:, i], label=f"DMP {labels[i]}", color="tab:blue", linewidth=1.5)
        axs[i].plot(traj[:, i], label=f"DEMO {labels[i]}", color="tab:orange", linestyle="--", linewidth=1.5)
        axs[i].set_ylabel(labels[i])
        axs[i].legend(loc="best")
        axs[i].grid(True, linestyle="--", alpha=0.4)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

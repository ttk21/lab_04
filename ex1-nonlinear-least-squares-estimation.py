import numpy as np
import visgeom as vg
import matplotlib
import matplotlib.pyplot as plt
from pylie import SO3, SE3
from optim import gauss_newton

"""Example 1 - Nonlinear least squares estimation"""


class NoiseFreePointAlignmentBasedPoseEstimatorObjective:
    """Implements linearisation of the noise-agnostic objective function"""

    def __init__(self, x_w, x_o):
        if x_w.shape[0] != 3 or x_w.shape != x_o.shape:
            raise TypeError('Matrices with corresponding points must have same size')

        self.x_w = x_w
        self.x_o = x_o
        self.num_points = x_w.shape[1]

    def linearise(self, T_wo):
        A = np.zeros((3 * self.num_points, 6))
        b = np.zeros((3 * self.num_points, 1))
        T_wo_inv = T_wo.inverse()

        # Enter the submatrices from each measurement:
        for i in range(self.num_points):
            A[3 * i:3 * (i + 1), :] = T_wo_inv.jac_action_Xx_wrt_X(self.x_w[:, [i]]) @ T_wo.jac_inverse_X_wrt_X()
            b[3 * i:3 * (i + 1)] = (self.x_o[:, [i]] - T_wo_inv * self.x_w[:, [i]])

        return A, b, b.T.dot(b)


def main():
    # World box.
    points_w = vg.utils.generate_box()

    # True observer pose.
    true_pose_wo = SE3((SO3.rot_z(np.pi), np.array([[3, 0, 0]]).T))

    # Observed box.
    points_o = vg.utils.generate_box(pose=true_pose_wo.inverse().to_tuple())

    # Perturb observer pose and use as initial state.
    init_pose_wo = true_pose_wo + np.random.randn(6, 1)

    # Estimate pose in the world frame from point correspondences.
    model = NoiseFreePointAlignmentBasedPoseEstimatorObjective(points_w, points_o)
    x, cost, A, b = gauss_newton(init_pose_wo, model)

    # Visualize (press a key to jump to the next iteration).
    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create figure and axis.
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Plot box and true state
    vg.plot_pose(ax, true_pose_wo.to_tuple(), scale=1, alpha=0.4)
    vg.utils.plot_as_box(ax, points_w, alpha=0.4)

    # Plot initial state (to run axis equal first time).
    ax.set_title('Cost: ' + str(cost[0]))
    artists = vg.plot_pose(ax, x[0].to_tuple(), scale=1)
    artists.extend(vg.utils.plot_as_box(ax, x[0] * points_o))
    vg.plot.axis_equal(ax)
    plt.draw()

    while True:
        if plt.waitforbuttonpress():
            break

    # Plot iterations
    for i in range(1, len(x)):
        for artist in artists:
            artist.remove()

        ax.set_title('Cost: ' + str(cost[i]))
        artists = vg.plot_pose(ax, x[i].to_tuple(), scale=1)
        artists.extend(vg.utils.plot_as_box(ax, x[i] * points_o))
        plt.draw()
        while True:
            if plt.waitforbuttonpress():
                break


if __name__ == "__main__":
    main()

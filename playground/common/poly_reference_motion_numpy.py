import numpy as np
import pickle


class PolyReferenceMotion:
    """Looks up the recorded reference motion nearest a commanded (dx, dy, dtheta).

    The command grid does not need to be dense: this indexes the flat list of
    recorded entries by nearest command-space distance, rather than assuming
    every combination of dx, dy, and dtheta was recorded (that assumption held
    for the original 6x4x10 auto-generated grid, but not for a hand-picked,
    human-reviewed set like the eight bdx_inspired motions, which only vary
    one axis at a time).
    """

    def __init__(self, polynomial_coefficients: str):
        data = pickle.load(open(polynomial_coefficients, "rb"))
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.command_points = None
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None

        self.process(data)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        commands = []
        entries = []
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])

            if self.period is None:
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            coeffs = [v for v in data[name]["coefficients"].values()]
            commands.append((dx, dy, dtheta))
            entries.append(coeffs)

        self.command_points = np.array(commands)
        self.data_array = entries

        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):
        dx = np.clip(dx, self.dx_range[0], self.dx_range[1])
        dy = np.clip(dy, self.dy_range[0], self.dy_range[1])
        dtheta = np.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1])

        query = np.array([dx, dy, dtheta])
        dists = np.sum((self.command_points - query) ** 2, axis=1)
        return int(np.argmin(dists))

    def sample_polynomial(self, t, coeffs):
        ret = []
        for c in coeffs:
            ret.append(np.polyval(np.flip(c), t))

        return ret

    def get_reference_motion(self, dx, dy, dtheta, i):
        idx = self.vel_to_index(dx, dy, dtheta)
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        t = np.clip(t, 0.0, 1.0)  # safeguard
        ret = self.sample_polynomial(t, self.data_array[idx])
        return ret


if __name__ == "__main__":

    PRM = PolyReferenceMotion(
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    )
    vals = []
    select_dim = -1
    for i in range(PRM.nb_steps_in_period):
        vals.append(PRM.get_reference_motion(0.0, -0.05, -0.1, i)[select_dim])

    # plot
    import matplotlib.pyplot as plt
    import numpy as np

    ts = np.arange(0, PRM.nb_steps_in_period)
    plt.plot(ts, vals)
    plt.show()

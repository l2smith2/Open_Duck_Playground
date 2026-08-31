import numpy as np
import pickle


class PolyReferenceMotion:
    """Builds the reference motion for a commanded (dx, dy, dtheta).

    The command grid does not need to be dense: this indexes the flat list of
    recorded entries by command-space distance, rather than assuming every
    combination of dx, dy, and dtheta was recorded (that assumption held for the
    original 6x4x10 auto-generated grid, but not for a hand-picked,
    human-reviewed set like the eight bdx_inspired motions, which only vary one
    axis at a time).

    Lookup interpolates between the two nearest recordings rather than snapping
    to one of them; see blend_for_command.
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
        self.data_array = np.array(entries)
        # dx is in m/s over roughly +-0.15 while dtheta is in rad/s over +-1.0,
        # so raw Euclidean distance in command space lets a physically trivial
        # yaw difference outweigh a large forward-speed difference. Measure
        # every axis relative to the span this reference actually covers.
        spans = self.command_points.max(axis=0) - self.command_points.min(axis=0)
        self.command_scale = 1.0 / np.where(spans > 1e-9, spans, 1.0)

        print("[Poly ref data] Done processing")

    def _clipped_query(self, dx, dy, dtheta):
        return np.array(
            [
                np.clip(dx, self.dx_range[0], self.dx_range[1]),
                np.clip(dy, self.dy_range[0], self.dy_range[1]),
                np.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1]),
            ]
        )

    def _scaled(self, points):
        """Command-space coordinates with each axis normalised by its span."""
        return points * self.command_scale

    def vel_to_index(self, dx, dy, dtheta):
        query = self._scaled(self._clipped_query(dx, dy, dtheta))
        dists = np.sum((self._scaled(self.command_points) - query) ** 2, axis=1)
        return int(np.argmin(dists))

    def blend_for_command(self, dx, dy, dtheta):
        """Coefficients for this command, interpolated between two recordings.

        Nearest-neighbour lookup turns the reference into a staircase. With the
        eight hand-picked bdx motions the whole band from 0.037 to 0.111 m/s is
        served the same 0.074 m/s gait, so a 0.10 m/s command is asked to
        imitate a gait that translates at 74% of it while tracking_lin_vel asks
        for the full speed. The policy cannot satisfy both, and it settles below
        the command.

        So this blends the nearest recording with whichever second recording
        best closes the remaining gap: the query is projected onto the segment
        from the nearest entry to each candidate, and the candidate with the
        smallest leftover error wins. Picking the second-nearest entry instead
        would not work on a dense grid, where the two nearest commands usually
        differ along an axis the query does not need. Sampling is linear in the
        coefficients, so blending them is the same as blending the two sampled
        trajectories, and the partner is chosen from entries that were each
        reviewed.

        The candidate set includes the nearest entry itself, whose segment has
        zero length and contributes t = 0, so a query no blend can improve on
        falls back to plain nearest-neighbour rather than to something worse.
        """
        query = self._scaled(self._clipped_query(dx, dy, dtheta))
        points = self._scaled(self.command_points)
        dists = np.sum((points - query) ** 2, axis=1)
        first = int(np.argmin(dists))

        anchor = points[first]
        remainder = query - anchor
        segments = points - anchor
        lengths = np.sum(segments * segments, axis=1)
        safe = np.where(lengths > 0.0, lengths, 1.0)
        t = np.where(
            lengths > 0.0, np.clip(segments @ remainder / safe, 0.0, 1.0), 0.0
        )
        leftover = np.sum((remainder - t[:, None] * segments) ** 2, axis=1)
        second = int(np.argmin(leftover))

        weight = float(t[second])
        return (1.0 - weight) * self.data_array[first] + weight * self.data_array[second]

    def sample_polynomial(self, t, coeffs):
        ret = []
        for c in coeffs:
            ret.append(np.polyval(np.flip(c), t))

        return ret

    def get_reference_motion(self, dx, dy, dtheta, i):
        coeffs = self.blend_for_command(dx, dy, dtheta)
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        t = np.clip(t, 0.0, 1.0)  # safeguard
        ret = self.sample_polynomial(t, coeffs)
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

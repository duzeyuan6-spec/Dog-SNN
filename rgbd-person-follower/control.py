"""Small forward-only controller; all stop paths bypass smoothing."""
import math


class FollowController:
    def __init__(self, config):
        self.c = config
        self.reset()

    def reset(self):
        self.vx = self.yaw = 0.0

    def update(self, distance, bearing, dt):
        c = self.c
        if (not all(math.isfinite(x) for x in (distance, bearing, dt))
                or distance <= 0 or distance > c["max_distance"] or dt <= 0):
            self.reset()
            return 0.0, 0.0
        dt = min(dt, 0.1)
        error = max(0.0, distance - c["target_distance"] - c["distance_deadband"])
        desired_vx = min(c["max_vx"], c["kp_distance"] * error)
        desired_vx *= max(0.0, 1.0 - abs(bearing) / c["turn_in_place_angle"])
        if distance < c["min_distance"]:
            desired_vx = 0.0
        desired_yaw = (0.0 if abs(bearing) < c["yaw_deadband"] else
                       max(-c["max_yaw"], min(c["max_yaw"], c["kp_yaw"] * bearing)))
        # ponytail: P + deadband is sufficient for visible targets at walking speed;
        # tune these measured limits before adding a more complex controller.
        if desired_vx == 0:
            self.vx = 0.0
        else:
            limit = c["acceleration"] if desired_vx > self.vx else c["deceleration"]
            self.vx += max(-limit * dt, min(limit * dt, desired_vx - self.vx))
        limit = c["yaw_acceleration"] * dt
        self.yaw += max(-limit, min(limit, desired_yaw - self.yaw))
        return self.vx, self.yaw

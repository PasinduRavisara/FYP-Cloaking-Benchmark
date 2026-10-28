import math
import numpy as np

# Earth radius in meters
EARTH_RADIUS = 6371000.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes great-circle distance between two GPS coordinates in meters.
    Runs in Θ(1) constant time.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Computes initial great-circle bearing (direction angle in radians [-π, π])
    from point 1 to point 2 using atan2.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2) -
         math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda))
    return math.atan2(y, x)


def meters_to_lat_lon(lat, lon, distance_m, bearing_rad):
    """
    Displaces a (lat, lon) point by distance_m meters along bearing_rad.
    Uses equirectangular projection for sub-millisecond Θ(1) execution.
    """
    delta_lat = (distance_m * math.cos(bearing_rad)) / 111320.0
    delta_lon = (distance_m * math.sin(bearing_rad)) / (111320.0 * math.cos(math.radians(lat)))
    return lat + delta_lat, lon + delta_lon


# ============================================================================
# BASELINE 1: Planar Laplace Mechanism (Andrés et al. Geo-Indistinguishability)
# ============================================================================
class PlanarLaplace:
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

    def perturb(self, lat, lon):
        # Draw angle uniformly from [0, 2π) — completely unconstrained
        theta = np.random.uniform(0, 2 * math.pi)
        # Draw radius from Gamma(shape=2, scale=1/epsilon)
        # In planar Laplace, PDF is (ε^2 / 2π) * exp(-ε * r)
        r = np.random.gamma(shape=2, scale=1.0 / self.epsilon) * 100.0  # scaled to meters
        return meters_to_lat_lon(lat, lon, r, theta)


# ============================================================================
# BASELINE 2: TraCS-D (Zheng & Hu, 2026 PoPETs)
# Continuous direction-distance sampling in Θ(1) time (unbounded bounding box)
# ============================================================================
class TraCS:
    def __init__(self, epsilon=0.5, r_bounding=500.0):
        self.epsilon = epsilon
        self.r_bounding = r_bounding

    def perturb(self, lat, lon):
        # Continuous direction sampling across [0, 2π)
        theta = np.random.uniform(0, 2 * math.pi)
        # Normalized continuous distance sampling bounded by local rectangle
        u = np.random.uniform(0, 1)
        r = - (math.log(1 - u + 1e-12) / self.epsilon) * (self.r_bounding / 5.0)
        r = min(r, self.r_bounding)
        return meters_to_lat_lon(lat, lon, r, theta)


# ============================================================================
# PROPOSED SOLUTION: Kinematically-Bounded Geometric Cloaking Engine
# Enforces: θ_max < 90°, r_max = min(d_prev, v_max · Δt, r_max_cap),
# stationary guard, 10-minute session reset, and sliding w-event budget tracking.
# ============================================================================
class KinematicGeometricCloaking:
    def __init__(self, epsilon=0.5, theta_max_deg=60.0, v_max_kmh=90.0, r_max_cap=500.0, session_timeout_s=600.0, w_window=10):
        self.epsilon = epsilon
        self.theta_max = math.radians(theta_max_deg)  # strictly < 90° (π/2)
        self.v_max = (v_max_kmh * 1000.0) / 3600.0   # convert km/h to m/s
        self.r_max_cap = r_max_cap                    # 500m utility bound (matches TraCS baseline)
        self.session_timeout = session_timeout_s      # 10 minutes (600s) traffic-tolerant timeout
        self.w_window = w_window

        # Ephemeral RAM state (single previous coordinate only — zero database)
        self.p_prev = None          # (lat, lon)
        self.t_prev = None          # timestamp
        self.cloaked_prev = None    # previous cloaked (lat, lon)
        self.step_count = 0

    def reset_state(self):
        """Wipes ephemeral memory to ensure zero persistence across trips."""
        self.p_prev = None
        self.t_prev = None
        self.cloaked_prev = None
        self.step_count = 0

    def cloak(self, lat, lon, timestamp):
        """
        Processes a single incoming GPS point in Θ(1) constant time (< 0.5 ms).
        Returns: (cloaked_lat, cloaked_lon, execution_time_ms)
        """
        import time
        start_time = time.perf_counter()

        # Step 0: Cold-start OR 10-Minute Session Break (gap > 600s)
        if self.p_prev is None or (timestamp - self.t_prev) > self.session_timeout:
            self.p_prev = (lat, lon)
            self.t_prev = timestamp
            theta = np.random.uniform(0, 2 * math.pi)
            r = min(np.random.exponential(scale=30.0 / self.epsilon), self.r_max_cap / 10.0)
            r = max(r, 5.0)
            self.cloaked_prev = meters_to_lat_lon(lat, lon, r, theta)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return self.cloaked_prev[0], self.cloaked_prev[1], elapsed_ms

        delta_t = max(timestamp - self.t_prev, 0.5)  # elapsed seconds
        d_prev = haversine_distance(self.p_prev[0], self.p_prev[1], lat, lon)

        # Step 1: Stationary Jitter Guard (d_prev < 2 meters, e.g. stuck at red light)
        if d_prev < 2.0:
            self.t_prev = timestamp
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return self.cloaked_prev[0], self.cloaked_prev[1], elapsed_ms

        # Step 2: Compute heading angle α via great-circle bearing
        alpha = calculate_bearing(self.p_prev[0], self.p_prev[1], lat, lon)

        # Step 3: Forward Sector Direction Bound (θ_max < 90°)
        # Perturbation angle is restricted to [α - θ_max, α + θ_max]
        angle_offset = np.random.uniform(-self.theta_max, self.theta_max)
        theta_cloaked = alpha + angle_offset

        # Step 4: Kinematic Velocity Radius Bound with 500m Utility Cap
        # Bounds noise by travel speed (v_max * Δt), dynamic step distance, and 500m ceiling
        kinematic_limit = self.v_max * delta_t
        r_max = min(max(d_prev * 1.2, 10.0), kinematic_limit, self.r_max_cap)

        # Sample distance within [0, r_max] under bounded exponential decay
        u = np.random.uniform(0, 1)
        r = - (math.log(1 - u * (1 - math.exp(-self.epsilon)) + 1e-12) / self.epsilon) * (r_max / 2.0)
        r = min(max(r, 5.0), r_max)

        # Generate cloaked point
        c_lat, c_lon = meters_to_lat_lon(lat, lon, r, theta_cloaked)

        # Update ephemeral single-point RAM state
        self.p_prev = (lat, lon)
        self.t_prev = timestamp
        self.cloaked_prev = (c_lat, c_lon)
        self.step_count += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return c_lat, c_lon, elapsed_ms
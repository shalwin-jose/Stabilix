class Predictor:

    def __init__(self):
        pass

    def predict(self, features):

        state = "Stable"
        risk = 2
        confidence = 95

        cpu = features["cpu_avg"]
        gpu = features["gpu_avg"]
        ram = features["ram_avg"]

        temp = features["gpu_temperature"]

        cpu_rate = features["cpu_perf_rate"]
        temp_rate = features["gpu_temp_rate"]

        # ------------------------------
        # Heavy CPU
        # ------------------------------

        if cpu > 85:

            state = "Heavy CPU Load"
            risk = 35
            confidence = 90

        # ------------------------------
        # Heavy GPU
        # ------------------------------

        if gpu > 90:

            state = "Heavy GPU Load"
            risk = 45
            confidence = 91

        # ------------------------------
        # Heavy RAM
        # ------------------------------

        if ram > 90:

            state = "High Memory Usage"
            risk = 40
            confidence = 90

        # ------------------------------
        # Thermal Stress
        # ------------------------------

        if temp > 82:

            state = "Thermal Stress"
            risk = 70
            confidence = 93

        # ------------------------------
        # Critical Temperature
        # ------------------------------

        if temp > 88:

            state = "Critical Temperature"
            risk = 90
            confidence = 97

        # ------------------------------
        # Rapid Heating
        # ------------------------------

        if temp_rate > 0.8:

            state = "Rapid Heating"
            risk += 15

        # ------------------------------
        # CPU Frequency Dropping
        # ------------------------------

        if cpu_rate < -2:

            risk += 10

        # ------------------------------
        # Clamp values
        # ------------------------------

        risk = max(0, min(risk, 100))
        confidence = max(0, min(confidence, 100))

        return {

            "state": state,
            "risk": risk,
            "confidence": confidence

        }
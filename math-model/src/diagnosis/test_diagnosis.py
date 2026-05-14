from src.diagnosis import ResidualObserver, DiagnosisStrategy

sensor_data = {
    "depth": 50.0,
    "target_z": 60.0,
    "true_depth": 50.0,
    "thruster": {
        "cmd_vz": 1.0,
        "actual_vz": 0.02,
        "current": 45.0,
    }
}

observer = ResidualObserver()
strategy = DiagnosisStrategy()

residuals = observer.compute(sensor_data)
result = strategy.diagnose(
    sensor_data=sensor_data,
    residuals=residuals,
    history=[sensor_data] * 20,
    ai_pred=0,
)

print("Residuals:", residuals)
print("Diagnosis:", result.as_dict())
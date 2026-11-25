"""Run in the isolated extracted source tree only (called with python -I -B)."""
from pathlib import Path
import hashlib, itertools, json, sys
import numpy as np
import torch

root, ref, output = map(lambda s: Path(s).resolve(), sys.argv[1:])
torch.set_num_threads(1)
sys.path[:0] = [str(root/'math-model/examples'), str(root/'math-model/src')]
import validate_six_dof_online_fusion as v
from demo_six_dof_unified_diagnostics import disturbance_provider
from sensors.sensor_faults import SensorFaultEvent, SensorFaultMode, SensorFaultInjector

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
reference=json.loads(ref.read_text(encoding='utf8'))
assert sha(v.DEFAULT_CHECKPOINT)==reference['ai_sha256']
if v.DEFAULT_TEMPORAL_CONFIG.exists():
    assert sha(v.DEFAULT_TEMPORAL_CONFIG)==reference['time_config_sha256']
bridge=v.SixDOFModelBridge(v.DEFAULT_CHECKPOINT,v.DEFAULT_TEMPORAL_CONFIG,v.AUVSixDOFMultiTaskDetector,device='cpu')
sup=v.FTCSafetySupervisor()
gate=v.AIControlEvidenceGate(v.AIEvidenceGateConfig(
    minimum_mode_probability=bridge.decision.temporal.config.enter_fault_probability,
    maximum_result_age_s=1.2*bridge.inference_stride*.05),sup.config)
sim=v.SixDOFSimulator(sensor_suite=v.SixDOFSensorSuite(seed=196112281,fault_injector=SensorFaultInjector(
    [SensorFaultEvent('dvl',SensorFaultMode.UNAVAILABLE,12.,18.)])),ftc_supervisor=sup,diagnostic_observer=bridge,ai_ftc_fusion=gate)
sim.reset(v.SixDOFState(position_ned=np.array([0.,0.,210.])))
logs=sim.run(46.,.05,v.mission_target(210),disturbance_provider)
assert len(logs)==920 and all(l['maintenance_diagnosis']['execution_mode']=='online_fusion' and 'ai_ftc_fusion' in l for l in logs)
assert all(a['ftc_action']==b['ftc_applied_action'] for a,b in zip(logs,logs[1:]))
for mod in (v,sys.modules[type(sim).__module__],sys.modules[type(sup).__module__],sys.modules[type(bridge).__module__],sys.modules[v.AUVSixDOFMultiTaskDetector.__module__]):
    assert Path(mod.__file__).resolve().is_relative_to(root)
def seq(fn): return [[round(next(g)['time'],3),x] for x,g in itertools.groupby(logs,key=fn)]
def phase(l): return l.get('mission_recovery',{}).get('phase')
def prefix(end):
    h=hashlib.sha256()
    for l in logs:
        if round(l['time'],6)>end:break
        for k in ('time','position_ned','commanded_thruster_forces','actual_thruster_forces'):
            h.update(np.asarray(l[k],dtype='<f8').tobytes())
        h.update(str((l['ftc_action'],phase(l),l['maintenance_diagnosis']['raw_mode_probabilities'])).encode())
    return h.hexdigest()
errors=[float(np.linalg.norm(l['estimated_position_ned'][:2]-l['position_ned'][:2])) for l in logs]
out=dict(frames=len(logs),actions=seq(lambda l:l['ftc_action']),guidance=seq(lambda l:l['ftc_guidance_mode_applied']),
         phases=seq(phase),ai_updates=sum(bool(l['maintenance_diagnosis']['updated']) for l in logs),
         ai_admissions=sum(bool(l['ai_ftc_fusion']['accepted']) for l in logs),
         isolations=sorted({l['ftc_targeted_thruster_name'] for l in logs if l['ftc_targeted_thruster_name']}),
         hashes=dict(pre_inj=prefix(11.95),pre_fix=prefix(20.20)),
         max_tilt_deg=max(float(np.degrees(np.arccos(np.clip(np.cos(l['euler_rpy'][0])*np.cos(l['euler_rpy'][1]),-1,1)))) for l in logs),
         final_xy_est_error_m=errors[-1])
out['differences']=[k for k,x in out.items() if not (abs(x-reference[k])<1e-8 if isinstance(x,float) else x==reference[k])]
out['environment']=dict(python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__)
output.write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps({k:out[k] for k in ('actions','phases','differences')}))

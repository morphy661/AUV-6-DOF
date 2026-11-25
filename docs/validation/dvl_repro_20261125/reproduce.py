"""Export the pinned committed runtime; never modifies the source checkout."""
import argparse, io, os, shutil, subprocess as sp, sys, tempfile, zipfile
from pathlib import Path
P=Path(__file__).resolve().parent
BASE='490f089d473be8f4be75429ec9d95d81aaceebf9'
MODEL='depth-sensor-fault-detection/depth_fault_detection'
CAL=MODEL+'/results/six_dof_hybrid_telemetry_0_300m_focus_v3_4_seed43_calibrated_20260903'
paths=['math-model/src','math-model/examples',MODEL+'/model_six_dof_multitask.py',
       MODEL+'/results/six_dof_v3_5_high_yaw_seed43_20260904/best_model.pth']
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--repo',type=Path,default=Path.cwd())
    a.add_argument('--mode',choices=['clean','config','nav','full'],default='full')
    a.add_argument('--output',type=Path,required=True)
    args=a.parse_args()
    repo=args.repo.resolve()
    git=['git','--git-dir='+str(repo/'.git')]
    listing=sp.check_output(git+['ls-tree','-r','--name-only',BASE,'--',CAL]).decode().splitlines()
    extra=[p for p in listing if p in [CAL+'/temporal_decision_config.json',CAL+'/maintenance_ticket_config.json']]
    archive=sp.check_output(git+['archive','--format=zip',BASE,'--',*paths,*extra])
    with tempfile.TemporaryDirectory(prefix='auv_dvl_repro_') as d:
        root=Path(d)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:z.extractall(root)
        if args.mode!='clean':
            (root/CAL).mkdir(parents=True,exist_ok=True)
            for f in (P/'config').iterdir():shutil.copyfile(f,root/CAL/f.name)
        for name in (['nav','diagnosis'] if args.mode=='full' else ['nav'] if args.mode=='nav' else []):
            sp.run(['git','apply','--no-index','--whitespace=nowarn',str(P/(name+'.patch'))],cwd=d,check=True)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(root/'mplcache'))
        output=args.output.resolve()
        sp.run([sys.executable,'-I','-B',str(P/'probe.py'),str(root),str(P/'reference.json'),str(output)],env=env,cwd=d,check=True)
        if args.mode=='full':
            import json
            if json.loads(output.read_text())['differences']:raise SystemExit('Reference mismatch')
if __name__=='__main__':main()

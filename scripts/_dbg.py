import subprocess, sys, os
cwd="/home/jpearce/Documents/Lucid"
r=subprocess.run([sys.executable,'-m','conceptpoc.cli','--config','configs/delta_pred.json','generate-delta','--prompt','The scientist discovered that','--steps','5','--delta','artifacts/delta_pred/delta_predictor.pt','--writer','artifacts/writer_p2_film/writer_cond.pt'],capture_output=True,text=True,cwd=cwd,env={**os.environ,'PYTHONPATH':'src'})
print('RC',r.returncode)
print('STDOUT repr>>>')
print(repr(r.stdout))
print('STDERR tail>>>')
print(repr(r.stderr[-400:]))

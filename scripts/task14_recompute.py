"""TASK14 post-hoc: the output-space λ that pure recomputation from tokens predicts for the R3 (corrupt) cell,
which keeps A's raw token at position t. Writes results14/recompute_prediction.json."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, numpy as np
from goalgeo import kvprior as KP, latentgoal as LG, filterstate as FS, beliefcausal as BC
env=LG.make_env("channel",4); TS={24:(2,4,8,16,23),64:(2,4,8,16,32,48,63)}
out={}
for T in (24,64):
    for t in TS[T]:
        XA,_=KP.lm_data(env,20000,T,seed=900+t); XB,_=KP.lm_data(env,20000,T,seed=950+t)
        JA=LG.filter_joint(env,XA)[:,t]; JB=LG.filter_joint(env,XB)[:,t]
        keep=np.flatnonzero(np.linalg.norm(FS.full(JA)-FS.full(JB),axis=1)>=2.0)[:1000]
        XA,XB,JA,JB=XA[keep],XB[keep],JA[keep],JB[keep]; x=XA[:,t+1]
        C=np.concatenate([XB[:,:t],XA[:,t:t+2]],1)          # B's tokens before t, A's x_t and x_{t+1}
        JC=LG.filter_joint(env,C)[:,t+1]
        oA=KP.centred_log(BC.ideal(env,KP.update(env,JA,x),"next_obs")); oB=KP.centred_log(BC.ideal(env,KP.update(env,JB,x),"next_obs"))
        oC=KP.centred_log(BC.ideal(env,JC,"next_obs"))
        lam,_=KP.along(oC,oA,oB); out[f"{T}_{t}"]=lam
        print(T,t,"pure-recomputation prediction for R3 (output λ):",round(lam,3))
json.dump(out,open("results14/recompute_prediction.json","w"),indent=1)

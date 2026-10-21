"""Frozen five-stage SNN recipe replicated from fresh seed-specific initialization."""
import sys,importlib
from common import config,Frames
seed=int(sys.argv[1]);stage=sys.argv[2]
assert seed in (43,44)
modules={'v2':'iterate_v2','v3':'iterate_v3','v4':'iterate','t2':'iterate_t2','t1':'iterate_t1'}
m=importlib.import_module(modules[stage])
original_config=config
m.config=lambda:dict(original_config(),seed=seed)
# Keep data splits, dataset ordering and fixed validation subset identical to seed42.
m.Frames=lambda cfg,split,limit=None:Frames(dict(cfg,seed=42),split,limit)
m.SPEC=dict(m.SPEC)
m.SPEC['version']='seed'+str(seed)+'/'+m.SPEC['version']
if 'initialization' in m.SPEC:m.SPEC['initialization']='seed'+str(seed)+'/'+m.SPEC['initialization']
if __name__=='__main__':m.main()

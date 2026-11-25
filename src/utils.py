"""Shared paths, reproducibility and plotting utilities for BRD4 QSAR."""
from pathlib import Path
import json, logging, hashlib, random
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs'
SEED = 42
STAGES = ['01_load_data','02_clean_data','03_feature_generation','04_train_test_split','05_feature_selection','06_lazypredict','07_optuna_tuning','08_final_model_evaluation']

def load_config():
    p=ROOT/'config.json'
    if not p.exists(): raise FileNotFoundError(f'Missing project config: {p}')
    return json.loads(p.read_text(encoding='utf-8'))

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)

def ensure_dirs():
    for stage in STAGES:
        for sub in ('tables','figures','models'): (OUTPUTS/stage/sub).mkdir(parents=True,exist_ok=True)

def stage_dir(stage):
    p=OUTPUTS/stage
    for sub in ('tables','figures','models'): (p/sub).mkdir(parents=True,exist_ok=True)
    return p

def save_joblib(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(obj,path)

def load_joblib(path): return joblib.load(path)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def write_manifest(stage, params, inputs=None, outputs=None):
    from datetime import datetime, timezone
    d=stage_dir(stage)
    hashes={str(Path(k)):sha256(k) for k in (inputs or []) if Path(k).is_file()}
    payload={'stage':stage,'seed':load_config().get('seed',SEED),'created_utc':datetime.now(timezone.utc).isoformat(),
             'parameters':params,'input_sha256':hashes,'outputs':[str(x) for x in (outputs or [])]}
    (d/'manifest.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

def setup_plotting():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':150,'savefig.dpi':1200,'font.size':12,'axes.titlesize':17,
      'axes.titleweight':'bold','axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':12,
      'ytick.labelsize':12,'lines.linewidth':2.8,'axes.spines.top':False,'axes.spines.right':False,
      'figure.facecolor':'white','axes.facecolor':'white'})

def save_figure(fig, base):
    base=Path(base); base.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(base.with_suffix('.png'),dpi=1200,bbox_inches='tight'); fig.savefig(base.with_suffix('.pdf'),bbox_inches='tight')

def print_versions():
    import sys, sklearn, pandas
    print('Python',sys.version.split()[0], '| pandas',pandas.__version__,'| scikit-learn',sklearn.__version__)
    try:
        from rdkit import rdBase; print('RDKit',rdBase.rdkitVersion)
    except Exception as e: print('RDKit unavailable:',e)

"""Stage 04: scaffold-group holdout split. Test identifiers are locked here."""
import json, pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from src.utils import OUTPUTS, load_config, stage_dir, setup_plotting, save_figure, write_manifest
import matplotlib.pyplot as plt

def scaffold_key(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None: return 'INVALID_'+str(smiles)
    sc=MurckoScaffold.GetScaffoldForMol(m)
    if sc.GetNumAtoms()==0: return 'ACYCLIC_'+Chem.MolToSmiles(m,canonical=True)
    return Chem.MolToSmiles(sc,canonical=True)

def run(input_path=None):
    cfg=load_config(); p=input_path or OUTPUTS/'02_clean_data/tables/cleaned_activity.csv'; df=pd.read_csv(p)
    df['group']=df.canonical_smiles.map(scaffold_key)
    # One deterministic group shuffle; approximate target proportion while guaranteeing group separation.
    splitter=GroupShuffleSplit(n_splits=1,test_size=cfg['test_size'],random_state=cfg['seed'])
    train_i,test_i=next(splitter.split(df,groups=df.group)); tr=df.iloc[train_i].copy(); te=df.iloc[test_i].copy()
    overlap=set(tr.group)&set(te.group)
    if overlap: raise RuntimeError(f'Scaffold leakage detected ({len(overlap)} groups)')
    d=stage_dir('04_train_test_split'); tr[['chembl_id','group']].to_csv(d/'tables/train_ids.csv',index=False); te[['chembl_id','group']].to_csv(d/'tables/test_ids.csv',index=False)
    summary=pd.DataFrame([{'n_total':len(df),'n_train':len(tr),'n_test':len(te),'train_fraction':len(tr)/len(df),'test_fraction':len(te)/len(df),'n_train_groups':tr.group.nunique(),'n_test_groups':te.group.nunique(),'scaffold_overlap':len(overlap),'seed':cfg['seed']}]); summary.to_csv(d/'tables/split_summary.csv',index=False)
    manifest={'seed':cfg['seed'],'test_size_requested':cfg['test_size'],'group_column':'Bemis-Murcko scaffold; acyclic molecules grouped by canonical connectivity','train_n':len(tr),'test_n':len(te),'train_groups':int(tr.group.nunique()),'test_groups':int(te.group.nunique()),'scaffold_overlap':0,'test_locked':True}
    (d/'models/split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    setup_plotting(); fig,ax=plt.subplots(figsize=(8,5)); ax.hist(tr.pIC50,bins=22,alpha=.65,label='Train',color='#0072B2'); ax.hist(te.pIC50,bins=22,alpha=.65,label='Locked test',color='#D55E00'); ax.set(xlabel='pIC50',ylabel='Count',title='Scaffold split activity distributions'); ax.legend(); save_figure(fig,d/'figures/train_test_activity'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(['Train','Test'],[tr.group.nunique(),te.group.nunique()],color=['#0072B2','#D55E00']); ax.set(ylabel='Unique scaffold groups',title=f'Scaffold overlap: {len(overlap)}'); save_figure(fig,d/'figures/scaffold_group_counts'); plt.close(fig)
    print(summary.to_string(index=False))
    write_manifest('04_train_test_split',manifest,[p],[d/'tables/train_ids.csv',d/'tables/test_ids.csv',d/'models/split_manifest.json'])
    return tr,te
if __name__=='__main__': run()

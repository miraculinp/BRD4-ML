"""Stage 01: validate source table and calculate pIC50."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.utils import ROOT, load_config, stage_dir, setup_plotting, save_figure, write_manifest

def run(data_path=None):
    cfg=load_config(); p=Path(data_path) if data_path else ROOT/cfg['data_file']
    if not p.is_absolute(): p=ROOT/p
    if not p.exists(): raise FileNotFoundError(f'Input data not found: {p}. Put data.csv in the project root.')
    df=pd.read_csv(p)
    required=[cfg['id_column'],cfg['smiles_column'],cfg['ic50_column']]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}; found {list(df.columns)}')
    out=pd.DataFrame({'chembl_id':df[cfg['id_column']].astype('string').str.strip(),
      'smiles':df[cfg['smiles_column']].astype('string').str.strip(),
      'ic50_nM':pd.to_numeric(df[cfg['ic50_column']],errors='coerce')})
    out['pIC50']=np.where(out.ic50_nM>0,9-np.log10(out.ic50_nM),np.nan)
    out['load_issue']=''
    out.loc[out.chembl_id.isna() | out.chembl_id.eq(''),'load_issue']+='missing_chembl_id;'
    out.loc[out.smiles.isna() | out.smiles.eq(''),'load_issue']+='missing_smiles;'
    out.loc[out.ic50_nM.isna(),'load_issue']+='non_numeric_or_missing_ic50;'
    out.loc[out.ic50_nM.le(0),'load_issue']+='nonpositive_ic50;'
    d=stage_dir('01_load_data'); out.to_csv(d/'tables/raw_activity.csv',index=False)
    setup_plotting(); fig,ax=plt.subplots(figsize=(8,5)); vals=out.pIC50.dropna()
    ax.hist(vals,bins=25,color='#0072B2',edgecolor='white'); ax.set(xlabel='pIC50',ylabel='Molecule count',title='Loaded activity distribution');
    save_figure(fig,d/'figures/pic50_histogram'); plt.close(fig)
    dup=int(out.chembl_id.duplicated().sum())
    print(f'Rows: {len(out)} | valid pIC50: {out.pIC50.notna().sum()} | duplicate IDs: {dup}')
    write_manifest('01_load_data',{'columns':required,'formula':'pIC50 = 9 - log10(IC50_nM)'},[p],[d/'tables/raw_activity.csv',d/'figures/pic50_histogram.png'])
    return out
if __name__=='__main__': run()

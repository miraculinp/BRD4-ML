"""Stage 02: RDKit standardisation and activity-aware duplicate consolidation."""
import pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from src.utils import stage_dir, setup_plotting, save_figure, write_manifest
import matplotlib.pyplot as plt
RDLogger.DisableLog('rdApp.warning')

def standardize(smiles, remove_salts=True):
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None: return None,None
    if remove_salts:
        try: mol=rdMolStandardize.LargestFragmentChooser().choose(mol)
        except Exception: pass
    try: mol=rdMolStandardize.Cleanup(mol); mol=rdMolStandardize.Uncharge(mol)
    except Exception: pass
    if mol is None or mol.GetNumAtoms()==0: return None,None
    return mol,Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)

def run(input_path=None, remove_salts=True):
    from src.utils import OUTPUTS
    p=input_path or OUTPUTS/'01_load_data/tables/raw_activity.csv'; df=pd.read_csv(p)
    records=[]; bad=[]
    for _,r in df.iterrows():
        mol,can=standardize(r.smiles,remove_salts)
        if mol is None or not np.isfinite(r.pIC50): bad.append({'chembl_id':r.chembl_id,'smiles':r.smiles,'reason':'invalid_structure_or_activity'}); continue
        records.append({'chembl_id':str(r.chembl_id),'smiles':r.smiles,'canonical_smiles':can,'pIC50':float(r.pIC50),'ic50_nM':float(r.ic50_nM)})
    valid=pd.DataFrame(records)
    merged=[]; logs=[]
    for smi,g in valid.groupby('canonical_smiles',sort=False):
        ids=sorted(set(g.chembl_id.astype(str))); med=float(g.pIC50.median())
        merged.append({'chembl_id':';'.join(ids),'chembl_ids':';'.join(ids),'canonical_smiles':smi,'pIC50':med,'n_measurements':len(g),'pIC50_sd':float(g.pIC50.std(ddof=1)) if len(g)>1 else 0.0})
        if len(g)>1: logs.append({'canonical_smiles':smi,'n_rows':len(g),'chembl_ids':';'.join(ids),'pIC50_values':';'.join(map(str,g.pIC50.tolist())),'median_pIC50':med})
    out=pd.DataFrame(merged); d=stage_dir('02_clean_data')
    out.to_csv(d/'tables/cleaned_activity.csv',index=False); pd.DataFrame(logs,columns=['canonical_smiles','n_rows','chembl_ids','pIC50_values','median_pIC50']).to_csv(d/'tables/deduplication_log.csv',index=False)
    pd.DataFrame(bad,columns=['chembl_id','smiles','reason']).to_csv(d/'tables/rejected_records.csv',index=False)
    setup_plotting(); fig,ax=plt.subplots(figsize=(7,5)); ax.bar(['Input valid rows','Standardised unique molecules'],[len(valid),len(out)],color=['#56B4E9','#009E73']); ax.set(ylabel='Count',title='Standardisation and deduplication'); save_figure(fig,d/'figures/counts_before_after'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(valid.pIC50,bins=25,alpha=.65,label='Before deduplication',color='#E69F00'); ax.hist(out.pIC50,bins=25,alpha=.65,label='After median consolidation',color='#0072B2'); ax.set(xlabel='pIC50',ylabel='Count',title='Activity distribution'); ax.legend(); save_figure(fig,d/'figures/pic50_before_after'); plt.close(fig)
    print(f'Valid rows: {len(valid)} | unique canonical structures: {len(out)} | rejected: {len(bad)} | duplicate groups: {len(logs)}')
    write_manifest('02_clean_data',{'remove_salts':remove_salts,'duplicate_rule':'median pIC50 per canonical isomeric SMILES'},[p],[d/'tables/cleaned_activity.csv',d/'tables/deduplication_log.csv'])
    return out
if __name__=='__main__': run()

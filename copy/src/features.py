"""Stage 03: chirality-aware ECFP4, physicochemical and optional Mordred descriptors."""
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from src.utils import OUTPUTS, load_config, stage_dir, setup_plotting, save_figure, write_manifest, save_joblib
import matplotlib.pyplot as plt
BASIC={'MolWt':Descriptors.MolWt,'MolLogP':Descriptors.MolLogP,'TPSA':rdMolDescriptors.CalcTPSA,'HBD':rdMolDescriptors.CalcNumHBD,'HBA':rdMolDescriptors.CalcNumHBA,'RotatableBonds':rdMolDescriptors.CalcNumRotatableBonds,'RingCount':Descriptors.RingCount,'HeavyAtomCount':Descriptors.HeavyAtomCount,'FractionCSP3':rdMolDescriptors.CalcFractionCSP3}

def run(input_path=None, include_mordred=True):
    cfg=load_config(); p=input_path or OUTPUTS/'02_clean_data/tables/cleaned_activity.csv'; df=pd.read_csv(p)
    radius=cfg['fingerprint_radius']; nbits=cfg['fingerprint_bits']; chir=cfg['use_chirality']
    # MorganGenerator is the current RDKit API and avoids the legacy Morgan deprecation warning.
    morgan_generator=rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nbits,includeChirality=chir)
    mols=[]; fps=[]; basic=[]; ids=[]; smiles=[]; failed3d=[]
    for _,r in df.iterrows():
        mol=Chem.MolFromSmiles(r.canonical_smiles)
        if mol is None: continue
        ids.append(r.chembl_id); smiles.append(r.canonical_smiles); mols.append(mol)
        fp=morgan_generator.GetFingerprint(mol); arr=np.zeros((nbits,),dtype=np.uint8); DataStructs.ConvertToNumpyArray(fp,arr); fps.append(arr)
        basic.append({f'phys_{name}':fn(mol) for name,fn in BASIC.items()})
    idx=pd.Index(ids,name='chembl_id'); matrices={'ecfp4':pd.DataFrame(fps,index=idx,columns=[f'ecfp4_{i}' for i in range(nbits)]),'physchem':pd.DataFrame(basic,index=idx)}
    if include_mordred:
        try:
            # `mordredcommunity` is installed from requirements.txt; its compatible Python API is `mordred`.
            from mordred import Calculator, descriptors
            calc=Calculator(descriptors,ignore_3D=True); m2=calc.pandas(mols); m2.index=idx; m2=m2.apply(pd.to_numeric,errors='coerce'); m2.columns=[f'mordred2d_{c}' for c in m2.columns]; matrices['mordred2d']=m2
        except Exception as e: print(f'Mordred 2D skipped: {e}')
        # 3D descriptor support is best effort; embed each conformer independently and preserve row alignment.
        try:
            # `mordredcommunity` is installed from requirements.txt; its compatible Python API is `mordred`.
            from mordred import Calculator, descriptors
            calc3=Calculator(descriptors,ignore_3D=False); mol3=[]; ok=[]
            for i,m in enumerate(mols):
                m3=Chem.AddHs(m); code=AllChem.EmbedMolecule(m3,randomSeed=cfg['seed'],maxAttempts=50)
                if code==0:
                    try: AllChem.MMFFOptimizeMolecule(m3,maxIters=150)
                    except Exception: pass
                    mol3.append(m3); ok.append(i)
                else: failed3d.append(ids[i])
            if mol3:
                m3df=calc3.pandas(mol3); m3df.index=idx[ok]; m3df=m3df.apply(pd.to_numeric,errors='coerce'); m3df.columns=[f'mordred3d_{c}' for c in m3df.columns]
                matrices['mordred3d']=m3df.reindex(idx)
        except Exception as e: print(f'Mordred 3D skipped: {e}')
    d=stage_dir('03_feature_generation')
    for family,mat in matrices.items():
        mat.to_parquet(d/f'tables/{family}_features.parquet'); save_joblib(mat,d/f'models/{family}_features.joblib')
        (d/f'tables/{family}_feature_names.txt').write_text('\n'.join(map(str,mat.columns)),encoding='utf-8')
    allx=pd.concat(matrices.values(),axis=1); allx=allx.loc[:,~allx.columns.duplicated()]
    allx.to_parquet(d/'tables/all_features.parquet'); save_joblib(allx,d/'models/all_features.joblib')
    y=df.set_index('chembl_id').reindex(allx.index)['pIC50']; target=y.to_frame('pIC50'); target.to_parquet(d/'tables/target.parquet')
    pd.DataFrame({'feature':allx.columns,'missing_fraction':allx.isna().mean().values}).sort_values('missing_fraction',ascending=False).to_csv(d/'tables/missing_value_summary.csv',index=False)
    (d/'tables/3d_embedding_failures.txt').write_text('\n'.join(failed3d),encoding='utf-8')
    setup_plotting(); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(list(matrices),[m.shape[1] for m in matrices.values()],color=['#0072B2','#E69F00','#009E73'][:len(matrices)]); ax.set(ylabel='Number of features',title='Feature family sizes'); save_figure(fig,d/'figures/feature_family_counts'); plt.close(fig)
    top=pd.read_csv(d/'tables/missing_value_summary.csv').head(30).sort_values('missing_fraction'); fig,ax=plt.subplots(figsize=(9,7)); ax.barh(top.feature,top.missing_fraction,color='#CC79A7'); ax.set(xlabel='Missing fraction',title='Features with most missing values'); save_figure(fig,d/'figures/top_feature_missingness'); plt.close(fig)
    print(f'Molecules: {len(allx)} | total features: {allx.shape[1]} | 3D embedding failures: {len(failed3d)}')
    write_manifest('03_feature_generation',{'radius':radius,'n_bits':nbits,'use_chirality':chir,'include_mordred':include_mordred},[p],[d/'tables/all_features.parquet',d/'tables/target.parquet'])
    return allx,y
if __name__=='__main__': run()

"""NumPy inference for target-specific UVE reflectance GPR/RC-GPR ensembles."""
import csv
import json
from pathlib import Path
import numpy as np


def load_weights(path):
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files if k!='metadata'}
        meta=json.loads(z['metadata'].item())
    n,d=a['support'].shape
    b=a['coefficients'].shape[0]
    expected={'wavelengths_nm':(d,),'feature_mask':(3,d),'feature_mean':(d,),
              'feature_scale':(d,),'target_mean':(3,),'target_scale':(3,),
              'lower':(3,),'upper':(3,),'amplitudes':(b,3),'length_scales':(b,3),
              'coefficients':(b,n,3),'baseline_coefficients':(b,n,3),'training_excel_rows':(n,)}
    for k,shape in expected.items():
        if a[k].shape!=shape or not np.isfinite(a[k]).all():raise ValueError('Invalid field '+k)
    if (a['feature_mask'].sum(axis=1)==0).any():raise ValueError('Empty target feature mask')
    for k in ['feature_scale','target_scale','amplitudes','length_scales']:
        if (a[k]<=0).any():raise ValueError('Nonpositive parameter '+k)
    return a,meta


def read_spectra(path,wavelengths):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('Empty input')
    keys={float(k):k for k in rows[0] if k!='sample_id'}
    ids=[r['sample_id'] for r in rows]
    X=np.array([[float(r[keys[float(w)]]) for w in wavelengths] for r in rows])
    if len(set(ids))!=len(ids) or not all(ids):raise ValueError('Invalid sample IDs')
    if not np.isfinite(X).all() or (X<=0).any():raise ValueError('Invalid reflectance')
    return ids,X


def predict_fits(X,a,baseline=False):
    z=(np.asarray(X)-a['feature_mean'])/a['feature_scale']
    coeff=a['baseline_coefficients'] if baseline else a['coefficients']
    q=np.empty((len(coeff),len(z),3))
    for j in range(3):
        mask=a['feature_mask'][j].astype(bool)
        dist=np.sum((z[:,None,mask]-a['support'][None,:,mask])**2,axis=2)
        kernel=a['amplitudes'][:,j,None,None]*np.exp(-dist[None]/(2*a['length_scales'][:,j,None,None]**2))
        q[:,:,j]=np.einsum('bni,bi->bn',kernel,coeff[:,:,j])
    t=q*a['target_scale']+a['target_mean']
    sig=np.exp(-np.logaddexp(0,-t))
    return a['lower']+(a['upper']-a['lower'])*sig


def predict(X,a,baseline=False):
    return predict_fits(X,a,baseline).mean(axis=0)



DEFAULT_WEIGHTS=Path(__file__).with_name('uve_rc_gpr_30_30_10.npz')

def self_test(weights_path):
    home=Path(__file__).parent
    a,meta=load_weights(weights_path)
    if a['support'].shape[0]!=108 or a['feature_mask'].sum(axis=1).tolist()!=[30,30,10] or len(a['coefficients'])!=1:
        raise ValueError('Unexpected model scope or feature counts')
    ids,X=read_spectra(home/'example_spectra.csv',a['wavelengths_nm'])
    with (home/'expected_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        expected={(r['model'],r['sample_id']):r for r in csv.DictReader(f)}
    for model,baseline in [('GPR',True),('RC-GPR',False)]:
        P=predict(X,a,baseline)
        for sid,p in zip(ids,P):
            actual=np.array([*p,p[0]/p[1],p[2]-p[0]/p[1]])
            row=expected[(model,sid)]
            reference=np.array([float(row[k]) for k in ['LNC','LKC','LNC_LKC','derived_ratio','consistency_error']])
            np.testing.assert_allclose(actual,reference,rtol=1e-8,atol=1e-8)
    print('PASS: 108 fitting samples, 30/30/10 feature masks, and both model predictions')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--baseline',action='store_true',help='Use paired independent-GPR coefficients')
    args=parser.parse_args()
    try:
        if args.self_test:
            self_test(args.weights)
        else:
            if args.input is None or args.output is None:
                parser.error('--input and --output are required unless --self-test is used')
            a,meta=load_weights(args.weights)
            ids,X=read_spectra(args.input,a['wavelengths_nm'])
            P=predict(X,a,args.baseline)
            with args.output.open('w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['sample_id','LNC','LKC','LNC_LKC','derived_ratio','consistency_error'])
                for sid,p in zip(ids,P):
                    derived=p[0]/p[1]
                    writer.writerow([sid,*[format(float(v),'.17g') for v in p],format(float(derived),'.17g'),format(float(p[2]-derived),'.17g')])
            print('Predicted',len(ids),'samples using the fixed 108-sample model')
    except (OSError,ValueError,KeyError,AssertionError) as exc:
        parser.error(str(exc))

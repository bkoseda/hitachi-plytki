import numpy as np
from .features import extract_features


def mix_noise(y,noise,snr_db,seed=42):
    y=np.asarray(y,float); noise=np.asarray(noise,float)
    if len(noise)==0 or not np.isfinite(noise).all():
        raise ValueError('Brak poprawnego tła.')
    rng=np.random.default_rng(seed)
    noise=np.tile(noise,int(np.ceil((len(y)+len(noise))/len(noise))))
    start=int(rng.integers(0,len(noise)-len(y)+1))
    noise=noise[start:start+len(y)]
    noise=noise-noise.mean()
    n_rms=np.sqrt(np.mean(noise**2))
    if n_rms<1e-10:
        raise ValueError('Tło jest ciszą.')
    scale=np.sqrt(np.mean(y**2))/(10**(snr_db/20)*n_rms)
    return (y+noise*scale).astype(np.float32)


def sensitivity_table(y,sr,noise=None,offset=.025,fmax=20000):
    import pandas as pd
    rng=np.random.default_rng(19)
    noise=rng.normal(size=len(y)) if noise is None else noise
    variants={'original':y,'gain -12 dB':y*.25,'noise 20 dB':mix_noise(y,noise,20),
              'noise 10 dB':mix_noise(y,noise,10),'noise 0 dB':mix_noise(y,noise,0)}
    rows=[]
    for name,signal in variants.items():
        features=extract_features(signal,sr,offset,fmax)
        rows.append(dict(variant=name,clipping_fraction=float(np.mean(np.abs(signal)>=.999)),**features))
    return pd.DataFrame(rows),variants

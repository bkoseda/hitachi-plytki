"""Synthetic resonators for software exercises ONLY; not real ceramic defect measurements."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import soundfile as sf


def resonator(sr=48000,duration=1.2,f0=1700,tau=.23,seed=0):
    rng=np.random.default_rng(seed)
    t=np.arange(round(sr*duration))/sr
    y=np.zeros_like(t)
    for i,ratio in enumerate([1,1.63,2.37,3.14]):
        y+=(1/(i+1))*np.sin(2*np.pi*f0*ratio*t+rng.uniform(-.1,.1))*np.exp(-t/(tau/(1+i*.2)))
    y+=rng.normal(0,.3,len(t))*np.exp(-t/.002)
    y*=np.minimum(t/.0005,1)
    return y/max(np.max(np.abs(y)),1e-9)


def generate_demo(output='data/demo',n_plates=30,seed=9):
    if n_plates<6:
        raise ValueError('Demo wymaga >=6 płytek.')
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    rows=[]; annotations={}
    sr=48000
    for p in range(n_plates):
        label='ok' if p < n_plates*2//3 else 'nok'
        plate=f'SYN_{p+1:03d}'
        session=f'S{p%3+1}'
        onsets=[.4,1.9,3.4]
        y=rng.normal(0,.0007,sr*5)
        f0=rng.normal(1700 if label=='ok' else 1490,75)
        tau=rng.uniform(.19,.32) if label=='ok' else rng.uniform(.065,.13)
        for h,t in enumerate(onsets):
            hit=resonator(sr,1.3,f0*rng.uniform(.985,1.015),tau*rng.uniform(.9,1.1),seed+p*10+h)
            start=round(t*sr)
            y[start:start+len(hit)]+=hit*rng.uniform(.35,.7)
        name=f'{plate}_{label}.wav'
        sf.write(out/name,y,sr,subtype='PCM_24')
        rows.append(dict(path=name,plate_id=plate,label=label,session_id=session,
                         position='A/B/C',environment='synthetic',notes='SYMULACJA, nie ceramika klienta',photo=''))
        annotations[name]=onsets
    sf.write(out/'background.wav',rng.normal(0,.002,sr*5),sr,subtype='PCM_24')
    rows.append(dict(path='background.wav',plate_id='BACKGROUND',label='background',session_id='S1',
                     position='',environment='synthetic',notes='Syntetyczny szum do testu',photo=''))
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
    (out/'annotations.json').write_text(json.dumps(annotations,indent=2))
    (out/'README.txt').write_text('DANE SYNTETYCZNE. Tłumione sinusoidy o celowo różnych parametrach.\n'
                                'Służą do ćwiczeń, testów kodu i UI. Wyniki nie dowodzą wykrywania wad ceramiki.\n')
    return out/'manifest.csv'

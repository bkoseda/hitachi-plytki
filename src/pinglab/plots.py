"""Shared plots for notebooks and standalone offline reports."""
import numpy as np
import pandas as pd
import librosa
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from .features import decay_curve
from .learning import plate_table

COLORS=['#117c83','#cc6735','#6555a4','#667788']


def theme(fig,title=None,height=420):
    fig.update_layout(template='plotly_white',height=height,font=dict(family='Arial, sans-serif',size=13),
                      margin=dict(l=60,r=25,t=60,b=50),colorway=COLORS,
                      title=title,legend=dict(orientation='h',y=1.13,x=0))
    return fig


def inspection(audio,onsets=(),n_fft=2048):
    y,sr=audio.y,audio.sr
    hop=max(sr//200,1)
    # Overview bins preserve BOTH min and max: a narrow transient must not vanish in decimation.
    step=max(1,len(y)//6000)
    count=len(y)//step
    blocks=y[:count*step].reshape(count,step)
    t=(np.arange(count)*step)/sr
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12,
                      subplot_titles=('Sygnał: kliknij i przybliż uderzenie','Spektrogram: jasne = więcej energii'))
    fig.add_trace(go.Scatter(x=t,y=blocks.min(axis=1),line=dict(width=.6,color=COLORS[0]),
                            name='min / max',showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=t,y=blocks.max(axis=1),line=dict(width=.6,color=COLORS[0]),
                            showlegend=False),row=1,col=1)
    fft=min(n_fft,len(y))
    s=np.abs(librosa.stft(y,n_fft=fft,hop_length=hop))
    db=librosa.amplitude_to_db(s,ref=np.max,top_db=80)
    # Bound HTML payload while keeping fine time resolution when zooming in a short event.
    stride=max(1,db.shape[1]//1200)
    f=librosa.fft_frequencies(sr=sr,n_fft=fft)
    mask=f<=min(sr/2,audio.native_sr/2,20000)
    fig.add_trace(go.Heatmap(x=librosa.frames_to_time(np.arange(s.shape[1]),sr=sr,hop_length=hop)[::stride],
                            y=f[mask],z=db[mask,::stride],colorscale='Magma',zmin=-80,zmax=0,
                            colorbar=dict(title='dB rel.',len=.4,y=.22)),row=2,col=1)
    for i,onset in enumerate(onsets):
        fig.add_vline(x=float(onset),line_color=COLORS[1],line_dash='dot',row='all',col=1)
        fig.add_annotation(x=float(onset),y=1,yref='y domain',text=str(i+1),showarrow=False,row=1,col=1)
    fig.update_yaxes(title_text='Amplituda',row=1,col=1)
    fig.update_yaxes(title_text='Częstotliwość [Hz]',row=2,col=1)
    fig.update_xaxes(title_text='Czas [s]',row=2,col=1)
    return theme(fig,audio.path.name,640)


def compare_hits(hits,sr=48000,names=None,offsets=None,fmax=20000):
    names=names or [f'Próbka {i+1}' for i in range(len(hits))]
    offsets=offsets or [.025]*len(hits)
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Widmo po ataku','Wybrzmiewanie'))
    for i,(y,name,offset) in enumerate(zip(hits,names,offsets)):
        ring=y[min(int((offset+.01)*sr),len(y)//2):]
        f,p=welch(ring,fs=sr,nperseg=min(4096,len(ring)))
        mask=(f>=80)&(f<=fmax)
        db=10*np.log10(np.maximum(p[mask],1e-20)/max(p[mask].max(),1e-20))
        fig.add_trace(go.Scatter(x=f[mask],y=db,name=name,line=dict(color=COLORS[i%4])),row=1,col=1)
        t,decay=decay_curve(y,sr,offset)
        fig.add_trace(go.Scatter(x=t,y=decay,name=name,showlegend=False,line=dict(color=COLORS[i%4])),row=1,col=2)
    fig.update_xaxes(title_text='Częstotliwość [Hz]',type='log',row=1,col=1)
    fig.update_yaxes(title_text='Moc [dB względem szczytu]',range=[-80,3],row=1,col=1)
    fig.update_xaxes(title_text='Czas od początku ogona [s]',row=1,col=2)
    fig.update_yaxes(title_text='Energia [dB względem maksimum]',range=[-70,3],row=1,col=2)
    return theme(fig,'Porównuj kształt, nie głośność',430)


def embedding_map(events,family='physics',color='label'):
    table,cols=plate_table(events,family)
    if len(table)<3:
        raise ValueError('PCA wymaga tu >=3 płytek. Dla pary użyj widm / tabeli różnic.')
    x=StandardScaler().fit_transform(table[cols])
    pca=PCA(n_components=2)
    z=pca.fit_transform(x)
    table['PC1'],table['PC2']=z[:,0],z[:,1]
    fig=px.scatter(table,x='PC1',y='PC2',color=color,symbol='label',hover_name='plate_id',
                   hover_data=['n_hits','session_id'],color_discrete_sequence=COLORS)
    fig.update_traces(marker=dict(size=11,line=dict(width=1,color='white')))
    variance=pca.explained_variance_ratio_.sum()*100
    return theme(fig,f'PCA · {family} · {variance:.0f}% zmienności w 2 osiach · obraz eksploracyjny'),table


def distance_map(events,family='physics'):
    table,cols=plate_table(events,family)
    x=StandardScaler().fit_transform(table[cols])
    d=cdist(x,x)/np.sqrt(len(cols))
    fig=go.Figure(go.Heatmap(z=d,x=table.plate_id,y=table.plate_id,colorscale='Viridis',
                            colorbar=dict(title='Odległość')))
    return theme(fig,'Podobieństwo płytek · mała odległość = podobne cechy',560)


def prediction_plot(predictions):
    p=predictions.sort_values('score_nok')
    fig=px.scatter(p,x='plate_id',y='score_nok',color='label',symbol='prediction',
                   hover_data=['n_hits','fold'],color_discrete_map={'ok':COLORS[0],'nok':COLORS[1]})
    fig.add_hline(y=.5,line_dash='dot',annotation_text='Próg roboczy 0.5')
    fig.update_traces(marker=dict(size=12))
    fig.update_yaxes(range=[-.03,1.03],title_text='Wynik NOK (niekalibrowany)')
    fig.update_xaxes(title_text='Płytka poza swoim treningiem')
    return theme(fig,'Każdy punkt to jedna fizyczna płytka')

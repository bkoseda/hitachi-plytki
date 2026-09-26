"""Build and execute a private, reproducible session notebook using pinglab primitives.

Run from the repository: .venv/bin/python scripts/run_badminton_session.py
Private audio and output notebooks stay in ignored directories; no client/demo config changes.
"""

import html
import json
import sys
import time
from pathlib import Path

import mistune
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/runs/badminton-four-series"
OUT.mkdir(parents=True, exist_ok=True)
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(display_name="Python (pinglab)", language="python", name="pinglab")
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""# Rakieta: centralne czy poza środkiem?
Cztery serie z jednej sesji. **Jedna rakieta, nie cztery niezależne obiekty.**

Notebook przechodzi przez zapis → kontrolę jakości → podział na uderzenia → widma i cechy → klasyfikację i odległość od wzorców → AST → odporność na szum.

**Run All** odtwarza analizę z lokalnego WAV. Nie nagrywa mikrofonu i nie wysyła dźwięku do sieci.
Etykiety dotyczą miejsca trafienia; wynik nie potwierdza wykrywania uszkodzeń płyt.
""")
code("""
from pathlib import Path
import sys, json, hashlib, io, base64
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/pinglab').exists())
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa, soundfile as sf
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display, Audio, HTML, Markdown
from pinglab.audio import load_audio, Config
from pinglab.recorder import segment_ringdown, ringdown_plot
from pinglab.features import extract_features, feature_columns, decay_curve
from pinglab.learning import make_classifier
from pinglab.series import evaluate_series, paired_folds
from pinglab.embeddings import ASTEncoder
from pinglab.robustness import mix_noise

CONFIG = ROOT / 'data/raw/badminton/session.json'
OUT = ROOT / 'reports/runs/badminton-four-series'
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.loads(CONFIG.read_text())
SOURCE = CONFIG.parent / cfg['audio']
RUN_AST = True  # gotowe wagi lokalne; bez treningu i bez pobierania
plt.rcParams.update({'figure.dpi':110, 'axes.spines.top':False, 'axes.spines.right':False})
colors = {'central':'#137f83', 'offcenter':'#c56838'}
r = load_audio(SOURCE, Config(sr=48000, max_seconds=180))
print('Plik:', SOURCE.name, '| SHA256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
display(pd.DataFrame([r.qc]).drop(columns=['warnings']))
print('Uwagi QC:', r.qc['warnings'] or 'brak')
""")
md("""## 1. Cztery serie, bez zapowiedzi w danych modelu
Pierwsza etykieta pochodzi z opisu nagrania. Trzy kolejne zapowiedzi potwierdzono lokalną transkrypcją: „uderzenia niecentralne, off center”, „druga seria uderzeń centralnych”, „druga seria uderzeń off center”.
Granice poniżej ustalono na przebiegu, przed modelowaniem. Odsłuchaj serie, jeśli chcesz sprawdzić trafienia.
""")
code("""
series = pd.DataFrame(cfg['series'])
display(series)
fig, ax = plt.subplots(figsize=(13,3))
step=max(1,len(r.y)//12000)
blocks=r.y[:len(r.y)//step*step].reshape(-1,step)
t=np.arange(len(blocks))*step/r.sr
ax.fill_between(t,blocks.min(1),blocks.max(1),color='#536574',lw=.3)
for s in cfg['series']:
 ax.axvspan(s['start_s'],s['end_s'],alpha=.13,color=colors[s['label']])
 ax.text((s['start_s']+s['end_s'])/2,.13,s['series_id'],ha='center',fontsize=9)
for a,b in cfg['speech_excluded_s']: ax.axvspan(a,b,color='grey',alpha=.3)
ax.set(xlabel='Czas całej sesji [s]',ylabel='Amplituda',title='Kolory: serie. Szare pola: wyłączone zapowiedzi.')
plt.show()
for s in cfg['series']:
 a,b=int(s['start_s']*r.sr),int(s['end_s']*r.sr)
 display(Markdown('**'+s['series_id']+'**'))
 display(Audio(r.y[a:b],rate=r.sr,normalize=False))
 sf.write(OUT/(s['series_id']+'.wav'),r.y[a:b],r.sr,subtype='PCM_24')
""")
md("""## 2. Uderzenie i wybrzmienie
Detektor szuka nagłych zmian widma wyłącznie wewnątrz serii. Nie wymuszamy dziesięciu wykryć.
Czerwony pas to umowne pierwsze 30 ms, zielony — obserwowany spadek do progu tła, pomarańczowy — brak takiego spadku przed następnym zdarzeniem. **To heurystyka, nie pomiar fizycznego czasu tłumienia.** Zmiana tła również może dać pomarańczową flagę.

Model dostaje jednakowe okno: 25 ms przed początkiem + 350 ms po nim, skrócone przed kolejnym uderzeniem. Nie uczymy modelu na długości ręcznie wyciętego klipu.
""")
code("""
rows=[]; clips={}; series_traces={}
for s in cfg['series']:
 z=r.y[int(s['start_s']*r.sr):int(s['end_s']*r.sr)]
 tab,t,db=segment_ringdown(z,r.sr,**cfg['detector'])
 series_traces[s['series_id']]=(z,tab,t,db)
 interactive=ringdown_plot(z,r.sr,tab,t,db)
 interactive.write_html(OUT/(s['series_id']+'_segmentacja.html'),include_plotlyjs=True)
 fig,axes=plt.subplots(2,1,figsize=(13,4.5),sharex=True)
 axes[0].plot(np.arange(len(z))/r.sr,z,lw=.4,color=colors[s['label']])
 axes[1].plot(t,db,lw=.7,color='#536574')
 for k,row in enumerate(tab.itertuples()):
  onset=float(row.onset_s);absolute=s['start_s']+onset
  a=max(0,onset-cfg['pre_s'])
  nxt=float(tab.iloc[k+1].onset_s) if k+1<len(tab) else len(z)/r.sr
  b=min(onset+cfg['feature_tail_s'],nxt-.015,len(z)/r.sr)
  clip=z[int(a*r.sr):int(b*r.sr)]
  eid=f"{s['series_id']}_{k+1:02d}"
  clips[eid]=clip
  uncertain=any(abs(absolute-v)<.05 for v in cfg['uncertain_onsets_s'])
  rows.append(dict(event_id=eid,series_id=s['series_id'],round=s['round'],label=s['label'],
   onset_s=absolute,offset_s=onset-a,window_s=len(clip)/r.sr,
   ringdown_s=row.duration_s,ringdown_status=row.status,uncertain=uncertain,
   peak_dbfs=20*np.log10(max(abs(clip).max(),1e-12)),
   **extract_features(clip,r.sr,onset-a,cfg['fmax'])))
  shade='#c47817' if 'przed zanikiem' in row.status else '#168b75'
  for ax in axes:
   ax.axvspan(onset,row.attack_end_s,color='#b4423e',alpha=.35)
   ax.axvspan(row.attack_end_s,row.end_s,color=shade,alpha=.22)
  axes[0].text(onset,abs(z).max()*1.1,str(k+1),fontsize=8,ha='center')
  axes[1].hlines(row.threshold_dbfs,onset,row.end_s,color=shade,lw=1.3)
 axes[0].set(title=s['series_id'],ylabel='Amplituda',ylim=(-abs(z).max()*1.25,abs(z).max()*1.4))
 axes[1].set(xlabel='Czas od początku serii [s]',ylabel='RMS [dBFS]',ylim=(-100,-15))
 plt.tight_layout();plt.show()
events=pd.DataFrame(rows)
events.to_csv(OUT/'features.csv',index=False)
display(events.groupby(['series_id','label'],sort=False).agg(uderzenia=('event_id','size'),
 szczyt_mediana_dbfs=('peak_dbfs','median'),niepewne=('uncertain','sum')))
display(events[['event_id','onset_s','ringdown_s','ringdown_status','uncertain']])
assert all(not (a<=v<=b) for v in events.onset_s for a,b in cfg['speech_excluded_s'])
assert events.window_s.min()>.05
""")
md("""## 3. Odsłuch i widma
Każde zdarzenie zachowuje czas w całym nagraniu. Etykieta pochodzi z serii; nie potwierdza, że każde pojedyncze trafienie rzeczywiście było centralne lub niecentralne.
Zdarzenie około 77 s jest znacznie słabsze i pozostaje oznaczone jako niepewne. Poniżej można odsłuchać wszystkie wykrycia.
""")
code("""
audio_rows=[]
for row in events.itertuples():
 signal=clips[row.event_id]
 buffer=io.BytesIO();sf.write(buffer,signal,r.sr,format='WAV',subtype='PCM_16')
 audio_rows.append(f'<tr><td>{row.event_id}</td><td>{row.onset_s:.3f} s</td><td>{"do sprawdzenia" if row.uncertain else "etykieta serii"}</td><td><audio controls preload="none" src="data:audio/wav;base64,{base64.b64encode(buffer.getvalue()).decode()}"></audio></td></tr>')
display(HTML('<details><summary>Odsłuch wszystkich uderzeń</summary><table>'+''.join(audio_rows)+'</table></details>'))
fig,axes=plt.subplots(2,2,figsize=(13,7))
for ax,s in zip(axes.flat,cfg['series']):
 ids=events.loc[events.series_id==s['series_id'],'event_id'].tolist()
 clip=clips[ids[len(ids)//2]]
 spec=librosa.amplitude_to_db(abs(librosa.stft(clip,n_fft=1024,hop_length=110)),ref=np.max)
 ax.imshow(spec,origin='lower',aspect='auto',extent=[-.025,len(clip)/r.sr-.025,0,r.sr/2],vmin=-70,vmax=0,cmap='magma')
 ax.set(title=ids[len(ids)//2]+' — dB względem maksimum',xlabel='Czas od uderzenia [s]',ylabel='Hz',ylim=(0,12000))
plt.tight_layout();plt.show()
fig,axes=plt.subplots(1,2,figsize=(13,4))
for s in cfg['series']:
 spectra=[];decays=[]
 for eid in events.loc[events.series_id==s['series_id'],'event_id']:
  clip=clips[eid];f,p=welch(clip[int(.035*r.sr):],fs=r.sr,nperseg=4096)
  spectra.append(p/max(p.sum(),1e-20))
  tt,d=decay_curve(clip,r.sr);decays.append(d)
 med=np.median(spectra,axis=0)
 axes[0].plot(f,10*np.log10(med+1e-15),label=s['series_id'],color=colors[s['label']],ls='-' if s['round']==1 else '--')
 common=min(map(len,decays))
 axes[1].plot(tt[:common],np.median([d[:common] for d in decays],axis=0),label=s['series_id'],color=colors[s['label']],ls='-' if s['round']==1 else '--')
axes[0].set(xlim=(100,15000),xlabel='Hz',ylabel='Względna energia [dB]',title='Mediana widma każdej serii');axes[0].set_xscale('log')
axes[1].set(xlabel='Czas po początku analizowanego ogona [s]',ylabel='dB względem maksimum',title='Mediana obwiedni zaniku');axes[1].legend()
plt.tight_layout();plt.show()
""")
md("""## 4. Reprezentacja cech
Porównujemy rezonans (3 cechy), cechy fizyczne (15), MFCC i mel. Amplituda jest normalizowana przed ekstrakcją; osobno sprawdzamy, ile można odgadnąć z samej głośności.
PCA poniżej służy tylko do oglądania danych. Zostało dopasowane do wszystkich punktów i **nie jest testem skuteczności**. Kształt punktu oznacza rundę.
""")
code("""
display(events.groupby('series_id')[['peak_hz','centroid_hz','decay10_s','peak_dbfs']].agg(['median','std']).round(2))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,family in zip(axes,['resonance','physics','mfcc']):
 x=StandardScaler().fit_transform(events[feature_columns(events,family)])
 xy=PCA(2).fit_transform(x)
 for s in cfg['series']:
  mask=events.series_id==s['series_id']
  ax.scatter(xy[mask,0],xy[mask,1],label=s['series_id'],c=colors[s['label']],marker='o' if s['round']==1 else '^')
 ax.set(title=family,xlabel='PC1',ylabel='PC2')
axes[-1].legend(fontsize=8);plt.tight_layout();plt.show()
""")
md("""## 5. Klasyfikacja i odległość od centralnych wzorców
Dwie próby: uczymy na seriach 1–2 i sprawdzamy 3–4, potem odwrotnie. Skalowanie i model uczą się tylko na części treningowej. Używamy ustalonej regresji logistycznej (`C=0.1`), bez poszukiwania najlepszego parametru.

Odległość pyta: „jak daleko ten dźwięk leży od centralnych uderzeń treningowych?”. Próg pochodzi wyłącznie z odległości między wzorcami treningowymi, ale nie jest skalibrowany. AUC ocenia kolejność wyników bez progu; balanced accuracy uśrednia trafność obu klas.

**Dwie rundy jednej sesji nie wystarczają do wiarygodnej oceny generalizacji.** Uderzenia są skorelowane; nie liczymy przedziałów ufności tak, jakby były niezależnymi obiektami. Porównanie wielu metod jest eksploracyjne.
""")
code("""
metrics=[];predictions=[]
for family in ['loudness','resonance','physics','mfcc','mel']:
 for method in (['logistic'] if family=='loudness' else ['logistic','distance']):
  m,p,audit=evaluate_series(events,family,method)
  metrics.append(m);predictions.append(p)
(OUT/'fold_audit.json').write_text(json.dumps(audit,indent=2))
display(pd.DataFrame(audit))
metrics=pd.concat(metrics,ignore_index=True);predictions=pd.concat(predictions,ignore_index=True)
display(metrics[['family','method','test_round','n_test','balanced_accuracy','auc','central_recall','offcenter_recall']].round(3))
fig,axes=plt.subplots(1,2,figsize=(8,3))
for ax,row in zip(axes,metrics.query("family=='physics' and method=='logistic'").itertuples()):
 cm=np.array(row.confusion_matrix);ax.imshow(cm,cmap='Blues',vmin=0,vmax=14)
 for i in range(2):
  for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
 ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['central','offcenter'],yticklabels=['central','offcenter'],xlabel='Predykcja',ylabel='Etykieta serii',title=f'Test: runda {row.test_round}')
plt.tight_layout();plt.show()
errors=predictions.query("family=='physics' and method=='logistic' and label!=prediction")
display(errors.merge(events[['event_id','onset_s','uncertain']],on='event_id'))
""")
md("""## 6. Gotowy model audio: zamrożony AST
Wykorzystujemy istniejący encoder AudioSet bez zmiany jego wag. Tylko prosta regresja na reprezentacjach uczy się rozróżniać nasze etykiety. Wagi są lokalne, a wektory buforowane według zawartości audio i wersji modelu.
""")
code("""
if RUN_AST:
 encoder=ASTEncoder(ROOT/'models/ast',offline=True)
 cache=ROOT/'data/cache/badminton_ast';cache.mkdir(parents=True,exist_ok=True)
 vectors=[]
 for row in events.itertuples():
  signal=clips[row.event_id]
  key=hashlib.sha256(signal.tobytes()+str(r.sr).encode()+encoder.revision.encode()).hexdigest()
  target=cache/(key+'.npy')
  if target.exists():v=np.load(target,allow_pickle=False)
  else:v=encoder.encode(signal,r.sr);np.save(target,v)
  assert v.shape==(768,) and np.isfinite(v).all()
  vectors.append(v)
 ast= pd.concat([events.reset_index(drop=True),pd.DataFrame(vectors,columns=[f'ast_{i:03d}' for i in range(768)])],axis=1)
 m,p,_=evaluate_series(ast,'ast','logistic')
 metrics=pd.concat([metrics,m],ignore_index=True);predictions=pd.concat([predictions,p],ignore_index=True)
 display(m[['family','test_round','balanced_accuracy','auc','central_recall','offcenter_recall']].round(3))
 print('AST:',encoder.revision,'| embedding 768 | bez dostrajania wag')
else: print('AST pominięty ustawieniem RUN_AST=False')
""")
md("""## 7. Czy wynik zależy od szumu albo jednego zdarzenia?
Model trenuje na oryginalnych uderzeniach. W odłożonej rundzie zmieniamy głośność albo dodajemy syntetyczny biały szum o zadanym SNR. To test kontrolowany; nie zastępuje nagrań z fabryki. Osobno powtarzamy ocenę bez słabego, niepewnego zdarzenia około 77 s.
""")
code("""
robust=[]
for name,snr in [('original',None),('gain -12 dB',None),('white noise 20 dB',20),('white noise 10 dB',10)]:
 altered=events.copy()
 if name!='original':
  for i,row in enumerate(events.itertuples()):
   signal=clips[row.event_id]
   signal=signal*10**(-12/20) if snr is None else mix_noise(signal,np.random.default_rng(100+i).normal(size=len(signal)),snr,seed=i)
   feats=extract_features(signal,r.sr,row.offset_s,cfg['fmax'])
   for key,value in feats.items():altered.loc[i,key]=value
 for family in ['resonance','physics','mfcc']:
  m,_,_=evaluate_series(events,family,test_features=altered)
  m['variant']=name;robust.append(m)
robust=pd.concat(robust,ignore_index=True)
display(robust.pivot_table(index=['family','variant'],columns='test_round',values='balanced_accuracy').round(3))
robust.to_csv(OUT/'robustness.csv',index=False)
uncertain=[]
for family in ['resonance','physics','mfcc','mel']:
 m,_,_=evaluate_series(events.loc[~events.uncertain].reset_index(drop=True),family)
 uncertain.append(m)
uncertain=pd.concat(uncertain,ignore_index=True)
display(uncertain[['family','test_round','n_test','balanced_accuracy','auc']].round(3))
uncertain.to_csv(OUT/'without_uncertain_event.csv',index=False)
""")
md("""## 8. Czy kilka uderzeń daje stabilniejszy wynik?
Dla ustalonego modelu fizycznego liczymy odsetek poprawnych decyzji opartych na medianie wyników 1, 3 lub 5 kolejnych uderzeń tej samej odłożonej serii. Okna zachodzą na siebie — to ilustracja stabilności, nie nowe niezależne testy.
""")
code("""
base=predictions.query("family=='physics' and method=='logistic'")
taps=[]
for sid,g in base.groupby('series_id',sort=False):
 score=g.score.to_numpy(); truth=g.label.iloc[0]=='offcenter'
 for n in [1,3,5]:
  decisions=[np.median(score[i:i+n])>=.5 for i in range(len(score)-n+1)]
  taps.append(dict(series_id=sid,n_taps=n,n_windows=len(decisions),fraction_correct=np.mean(np.array(decisions)==truth)))
display(pd.DataFrame(taps).pivot(index='series_id',columns='n_taps',values='fraction_correct').round(3))
pd.DataFrame(taps).to_csv(OUT/'multiple_hits.csv',index=False)
""")
md("""## 9. Zapis wyników i modelu roboczego
Zapisujemy stały wariant „cechy fizyczne + regresja” na wszystkich oznaczonych zdarzeniach. Jego trafności treningowej nie podajemy jako wyniku testu. Do kolejnej oceny potrzebna będzie nowa seria, najlepiej po ponownym ustawieniu stanowiska.

W katalogu wyników znajdziesz też cztery WAV-y serii, CSV cech i prognoz oraz interaktywne wykresy segmentacji. Surowy plik i notebook tej sesji nie wchodzą do paczki demonstracyjnej dla innych osób.
""")
code("""
import joblib
cols=feature_columns(events,'physics')
model=make_classifier('physics').fit(events[cols],(events.label=='offcenter').astype(int))
joblib.dump(model,OUT/'exploratory_physics.joblib')
contract=dict(classes=['central','offcenter'],columns=cols,sr=r.sr,pre_s=cfg['pre_s'],
 feature_tail_s=cfg['feature_tail_s'],fmax=cfg['fmax'],training_events=events.event_id.tolist(),
 source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 scope='one racket, same session, exploratory; not defect prediction',threshold=.5)
(OUT/'model_contract.json').write_text(json.dumps(contract,indent=2))
metrics.to_json(OUT/'metrics.json',orient='records',indent=2)
predictions.to_csv(OUT/'predictions.csv',index=False)
print('Podsumowanie: średnia balanced accuracy dwóch rund (opisowo, nie ranking do wdrożenia)')
display(metrics.groupby(['family','method'])[['balanced_accuracy','auc']].mean().round(3))
print('GOTOWE — rzeczywista sesja badminton; wynik eksploracyjny jednej rakiety.')
""")
nb.cells = cells
path = OUT / "Badminton_eksploracja.ipynb"
start = time.perf_counter()
render_only = "--render-only" in sys.argv
if render_only:
    nb = nbf.read(path, as_version=4)
else:
    (OUT / "validation.json").write_text(json.dumps({"status": "running"}))
    print("Executing", path, flush=True)
    try:
        NotebookClient(
            nb, timeout=1200, kernel_name="pinglab", resources={"metadata": {"path": str(ROOT)}}
        ).execute()
    except Exception:
        (OUT / "validation.json").write_text(json.dumps({"status": "failed"}))
        raise
    finally:
        nbf.write(nb, path)
assert all(c.get("execution_count") is not None for c in nb.cells if c.cell_type == "code")
assert not any(o.output_type == "error" for c in nb.cells for o in c.get("outputs", []))
parts = [
    """<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Badminton — cztery serie</title><style>body{max-width:1120px;margin:40px auto;padding:0 24px;background:#faf9f6;color:#23303d;font:16px/1.6 system-ui}h1{font-size:36px}h2{margin-top:48px;border-top:1px solid #cdd7dc;padding-top:20px}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #d9e0e4}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f2;padding:12px;font-size:12px}.output{overflow-x:auto;margin:12px 0}details{margin:8px 0}summary{cursor:pointer;background:#e8eeee;padding:8px}audio{max-width:100%}</style><body>"""
]
summary_rows = []
records = json.loads((OUT / "metrics.json").read_text())
for family, title in [
    ("physics", "Cechy fizyczne + regresja"),
    ("mfcc", "MFCC + regresja"),
    ("ast", "AST + regresja"),
]:
    values = sorted(
        [r for r in records if r["family"] == family and r["method"] == "logistic"],
        key=lambda r: r["test_round"],
    )
    if values:
        summary_rows.append(
            "<tr><td>"
            + title
            + "</td>"
            + "".join(f"<td>{v['balanced_accuracy']:.1%}</td>" for v in values)
            + "</tr>"
        )
summary = (
    """<section style="border-left:4px solid #c47817;padding:12px 20px;background:#fff4e5"><h2 style="margin-top:0;border:0;padding:0">Wynik tej sesji</h2><p>Nie uzyskaliśmy stabilnego rozdzielenia klas. Porównujemy całe odłożone serie, nie losowe uderzenia z tej samej serii.</p><table><tr><th>Reprezentacja</th><th>Test: pierwsza para</th><th>Test: druga para</th></tr>"""
    + "".join(summary_rows)
    + """</table><p>Procenty to średnia trafności obu klas (balanced accuracy; 50% odpowiada np. ciągłemu wybieraniu jednej klasy). Etykiety pochodzą z serii i wymagają potwierdzenia dla nietrafionych odbić.</p><p>W pierwszej parze serie wyraźnie różnią się głośnością; w drugiej ich mediany są zbliżone. To wskazuje na zmianę warunków lub siły odbić, nie dowodzi jej przyczyny. Najpierw warto przejrzeć wykrycia, potem zebrać nową parę serii.</p></section>"""
)
for index, c in enumerate(nb.cells):
    if c.cell_type == "markdown":
        parts.append(mistune.html(c.source))
        if index == 0:
            parts.append(summary)
            parts.append(
                "<p>Interaktywna segmentacja: "
                + " · ".join(
                    f'<a href="{sid}_segmentacja.html">{sid}</a>'
                    for sid in ["central_1", "offcenter_1", "central_2", "offcenter_2"]
                )
                + "</p>"
            )
        continue
    parts.append("<details><summary>Kod — rozwiń</summary><pre>" + html.escape(c.source) + "</pre></details>")
    for o in c.get("outputs", []):
        if o.output_type == "error":
            raise RuntimeError(str(o))
        d = o.get("data", {})
        if "image/png" in d:
            v = '<img alt="Wykres analizy" src="data:image/png;base64,' + d["image/png"] + '">'
        elif "text/html" in d:
            v = d["text/html"]
        elif "text/markdown" in d:
            v = mistune.html(d["text/markdown"])
        else:
            v = "<pre>" + html.escape(o.get("text", d.get("text/plain", ""))) + "</pre>"
        if "<table" in v and v.count("<tr") > 20 and not v.startswith("<details"):
            v = "<details><summary>Pełna tabela — rozwiń</summary>" + v + "</details>"
        parts.append('<div class="output">' + v + "</div>")
parts.append("</body></html>")
(OUT / "Badminton_raport.html").write_text("\n".join(parts))
if not render_only:
    (OUT / "validation.json").write_text(
        json.dumps(
            dict(
                status="passed",
                seconds=round(time.perf_counter() - start, 2),
                code_cells=sum(c.cell_type == "code" for c in cells),
                source="real badminton session",
            ),
            indent=2,
        )
    )
print("DONE", round(time.perf_counter() - start, 1), flush=True)

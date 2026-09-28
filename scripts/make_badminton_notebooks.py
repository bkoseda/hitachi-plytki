"""Build independent badminton variants from the reviewed session notebook."""

import copy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
master = nbformat.read(ROOT / "deliverables/Badminton_Ping_Lab/analiza.ipynb", 4)
DEST = ROOT / "notebooks/badminton"
DEST.mkdir(exist_ok=True)


def md(s):
    return nbformat.v4.new_markdown_cell(s)


def code(s):
    return nbformat.v4.new_code_cell(s)


def cells(*indices):
    return [copy.deepcopy(master.cells[i]) for i in indices]


IMPORTS = master.cells[1].source.split("import numpy as np", 1)[1]
SETUP = (
    """from pathlib import Path
import sys, json, hashlib, io, base64
from datetime import datetime
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/pinglab').is_dir()), None)
assert ROOT is not None, 'Otwórz notebook z folderu projektu ceramic-ping-lab.'
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
%matplotlib inline
cfg = json.loads((ROOT/'data/raw/badminton/session.json').read_text())
SOURCE = ROOT/'data/raw/badminton'/cfg['audio']
assert SOURCE.is_file(), 'Brak nagrania badminton: nie przełączamy się automatycznie na demo.'
OUT = ROOT/'reports/runs/badminton-notebooks'/NOTEBOOK/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
OUT.mkdir(parents=True)
import numpy as np"""
    + IMPORTS
)
SETUP = SETUP.replace("RUN_AST = True", "RUN_AST = False")

# Same reviewed detector and feature windows, without repeated plots in model notebooks.
QUIET = master.cells[5].source
start = QUIET.index(" fig,axes=")
end = QUIET.index(" for k,row", start)
QUIET = QUIET[:start] + QUIET[end:]
start = QUIET.index("  shade=")
end = QUIET.index("events=pd.DataFrame", start)
QUIET = QUIET[:start] + QUIET[end:]


def intro(number, title):
    return md(f"""# {number} · Badminton — {title}
**Dane rzeczywiste:** cztery serie central/off-center z jednej rakiety, 46 automatycznych kandydatów.
To demonstracja narzędzi, nie pomiar skuteczności wykrywania wad ceramiki.

Wybierz kernel `.venv` / Ping Lab i **Run All**. Ten notebook działa samodzielnie;
nie musisz wcześniej uruchamiać pozostałych. Granice serii i zapowiedzi pochodzą
z `data/raw/badminton/session.json`. `project.json` pozostaje konfiguracją oryginalnych notebooków.
Nowe wyniki trafiają do osobnego katalogu; nagranie źródłowe pozostaje bez zmian.
""")


def setup(name, ast=False):
    return code(f"NOTEBOOK = {name!r}\n" + SETUP.replace("RUN_AST = False", f"RUN_AST = {ast}"))


def prepared(name, title, ast=False):
    return [
        intro(name[:2], title),
        setup(name, ast),
        md(
            "## Przygotowanie uderzeń\nTen sam detektor i okna: 25 ms przed + 350 ms po początku, skrócone przed kolejnym zdarzeniem. Etykieta pochodzi z serii. Zdarzenie około 77 s pozostaje niepewne."
        ),
        code(QUIET),
    ]


def write(name, body):
    body.append(code("print('GOTOWE:', NOTEBOOK, '| pliki wynikowe:', OUT)"))
    n = nbformat.v4.new_notebook(
        cells=body,
        metadata={
            "kernelspec": {"display_name": "Ping Lab (Python)", "language": "python", "name": "pinglab"},
            "language_info": {"name": "python"},
        },
    )
    for c in n.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
    nbformat.write(n, DEST / (name + ".ipynb"))


name = "00_eksploracja_do_przegladu"
body = [intro("00", "przegląd całej sesji"), setup(name)] + cells(2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
body += [
    md(
        "## Wniosek do pokazania\nW pierwszej serii centralnej widać zmianę głośności między rundami. Proste modele mają niestabilne wyniki. Nie utożsamiamy pojedynczych uderzeń z niezależnymi rakietami. Kolejne widoki: 01 — para A/B, 02 — ręczna korekta, 04 — AST, 05 — szum, 06 — porównanie modeli, 07 — interaktywne wybrzmienia."
    )
]
write(name, body)

name = "01_pierwsza_para"
body = prepared(name, "odsłuch A/B i pojedyncze uderzenia")
body += [
    md(
        "## Wybierz parę\nMożesz zmienić ID i uruchomić tę komórkę ponownie. Pierwszy odsłuch zachowuje głośność, drugi wyrównuje szczyty, aby łatwiej porównać barwę."
    ),
    code("""EVENT_A = 'central_1_01'
EVENT_B = 'offcenter_1_01'
from pinglab.plots import compare_hits
selected = events.set_index('event_id').loc[[EVENT_A,EVENT_B]]
display(selected[['series_id','label','onset_s','peak_dbfs','uncertain']])
for normalize in [False,True]:
 display(Markdown('**Wyrównane szczyty**' if normalize else '**Oryginalna głośność**'))
 for eid in [EVENT_A,EVENT_B]:
  print(eid);display(Audio(clips[eid],rate=r.sr,normalize=normalize))
fig=compare_hits([clips[EVENT_A],clips[EVENT_B]],r.sr,[EVENT_A,EVENT_B],
                 selected.offset_s.tolist(),cfg['fmax'])
fig.show(renderer='plotly_mimetype')
fig.write_html(OUT/'porownanie_AB.html',include_plotlyjs=True)
keys=['peak_hz','centroid_hz','bandwidth_hz','decay10_s','late_early_ratio','flatness']
display(selected[keys].T.round(3))
"""),
    md(
        "**Do opowiedzenia:** czy różnica wynika z samej głośności, czy zostaje także po wyrównaniu poziomów? Wybrane dwa uderzenia ilustrują różnicę; nie są oceną klasyfikatora."
    ),
]
write(name, body)

name = "02_seria_i_uderzenia"
seg = master.cells[5].source.replace(
    "segment_ringdown(z,r.sr,**cfg['detector'])",
    "segment_ringdown(z,r.sr,CORRECTIONS.get(s['series_id']),**cfg['detector'])",
)
body = [intro("02", "segmentacja i ręczna korekta"), setup(name)] + cells(2, 3)
body += [
    md(
        "## Korekta detekcji\nSekundy liczymy od początku wybranej serii, nie całego nagrania. Brak klucza = automat. Lista `[]` = odrzucenie wszystkich wykryć tej serii. Korekty dotyczą tego uruchomienia; zapisujemy je w jego wynikach. Pozostałe notebooki nadal używają automatu, dopóki jawnie nie przeniesiemy korekt."
    ),
    code("""CORRECTIONS = {}  # np. {'central_1': [2.025,3.17,...]} — wpisz pełną listę rzeczywistych początków
known={s['series_id']:s for s in cfg['series']}
assert set(CORRECTIONS) <= set(known)
for sid,seconds in CORRECTIONS.items():
 assert all(np.isfinite(v) and 0<=v<known[sid]['end_s']-known[sid]['start_s'] for v in seconds)
 assert all(a<b for a,b in zip(seconds,seconds[1:])), 'Czasy muszą rosnąć.'
(OUT/'corrections.json').write_text(json.dumps(CORRECTIONS,indent=2))
"""),
    master.cells[4],
    code(seg),
    md(
        "## Przybliż wybraną serię\nNa wykresie możesz zaznaczyć prostokąt, aby przybliżyć fragment. Dwuklik resetuje widok."
    ),
    code("""SERIES_ID = 'central_1'
z,tab,t,db = series_traces[SERIES_ID]
ringdown_plot(z,r.sr,tab,t,db).show(renderer='plotly_mimetype')
"""),
]
write(name, body)

name = "03_cechy_i_modele"
body = prepared(name, "cechy, podobieństwo i proste modele") + cells(8, 9, 10, 11)
body += [
    md(
        "## Które uderzenia są podobne?\nMacierz odległości na wybranych cechach. To widok opisowy całego zbioru; skalowanie tutaj nie jest wykorzystywane do oceny modelu powyżej."
    ),
    code("""from scipy.spatial.distance import cdist
FAMILY='physics'
cols=feature_columns(events,FAMILY)
z=StandardScaler().fit_transform(events[cols])
d=cdist(z,z)/np.sqrt(len(cols))
fig,ax=plt.subplots(figsize=(9,7))
im=ax.imshow(d,cmap='viridis');fig.colorbar(im,ax=ax,label='Odległość w wystandaryzowanych cechach')
starts=events.groupby('series_id',sort=False).size()
centers=np.cumsum(starts)-starts/2-.5
ax.set(xticks=centers,yticks=centers,xticklabels=starts.index,yticklabels=starts.index,title=FAMILY)
plt.xticks(rotation=25);plt.tight_layout();plt.show()
"""),
    md(
        "**Granica interpretacji:** można ocenić przenoszenie między rundami tej samej rakiety. Test niezależnych detali i test nowej sesji są tutaj niemożliwe; nie tworzymy fikcyjnych plate_id."
    ),
]
write(name, body)

name = "04_gotowy_model_audio"
ast_code = master.cells[13].source.replace("offline=False", "offline=True")
body = prepared(name, "zamrożony AST i mapa embeddingów", True) + cells(10, 11)
body += [
    md(
        "## AST bez zmiany wag\nEncoder daje 768 liczb na uderzenie. Korzystamy z lokalnych wag i cache. Audio jest przeliczane do 16 kHz, więc AST nie wykorzystuje pasma powyżej 8 kHz. Przy nowej instalacji uruchom wcześniej `python scripts/warm_model.py`."
    ),
    code(ast_code),
    code("""ast.to_csv(OUT/'features_ast.csv',index=False)
from sklearn.preprocessing import Normalizer
z=Normalizer().fit_transform(ast[feature_columns(ast,'ast')])
xy=PCA(n_components=2).fit_transform(z)
fig,ax=plt.subplots(figsize=(8,5))
for sid,g in events.groupby('series_id',sort=False):
 ix=g.index;ax.scatter(xy[ix,0],xy[ix,1],c=colors[g.label.iloc[0]],marker='o' if g['round'].iloc[0]==1 else '^',label=sid)
ax.set(xlabel='PC1',ylabel='PC2',title='AST — PCA tylko do oglądania');ax.legend();plt.show()
display(metrics.query("method=='logistic'")[['family','test_round','balanced_accuracy','auc','central_recall','offcenter_recall']].round(3))
"""),
]
write(name, body)

name = "05_odpornosc_i_protokol"
body = prepared(name, "szum, głośność i protokół pomiaru")
body += [
    md(
        "## Posłuchaj kontrolowanej zmiany\nBiały szum jest syntetyczny, nie pochodzi z fabryki. Mniejszy SNR oznacza silniejszy szum."
    ),
    code("""from pinglab.robustness import sensitivity_table
from pinglab.plots import compare_hits
EVENT_ID='central_1_01'
offset=float(events.set_index('event_id').loc[EVENT_ID,'offset_s'])
stats,variants=sensitivity_table(clips[EVENT_ID],r.sr,offset=offset,fmax=cfg['fmax'])
display(stats[['variant','peak_hz','centroid_hz','decay10_s','flatness','clipping_fraction']].round(3))
for key in ['original','gain -12 dB','noise 20 dB','noise 10 dB']:
 print(key);display(Audio(variants[key],rate=r.sr,normalize=False))
compare_hits([variants['original'],variants['noise 10 dB']],r.sr,['Oryginał','Szum 10 dB'],[offset]*2,cfg['fmax']).show(renderer='plotly_mimetype')
"""),
] + cells(14, 15)
body += [
    md(
        "## Kolejne nagranie\nStały mikrofon, gain, podparcie i odległość. Nagrywaj obie klasy w każdej rundzie. Zbierz nową sesję po ponownym ustawieniu stanowiska. Najpierw seria z pełnym wybrzmieniem, potem szybsze trafienia. Ten test zmienia sygnał testowy; nie trenujemy tutaj na sztucznie zaszumionych kopiach i nie udajemy dodatkowych niezależnych obiektów."
    )
]
write(name, body)

name = "06_porownanie_metod"
body = prepared(name, "modele, błędy i liczba uderzeń", True) + cells(10, 11)
body += [
    md("## Opcjonalny AST\nLokalne wagi; RUN_AST można wyłączyć w pierwszej komórce."),
    code(ast_code),
] + cells(16, 17, 20, 21)
write(name, body)

name = "07_nagrywanie_i_wybrzmienia"
body = [
    intro("07", "interaktywne wybrzmienia i opcjonalny mikrofon"),
    setup(name),
    md(
        "## Wybierz serię lub nowe nagranie\nDomyślnie analizujemy central_1 z istniejącego nagrania. Mikrofon jest używany wyłącznie po zmianie `RECORD=True`. Cała sesja zawiera zapowiedzi głosowe, dlatego detektor dostaje wybraną serię bez mowy."
    ),
    code("""from pinglab.recorder import record_wav, export_pings
RECORD=False
SERIES_ID='central_1'
SECONDS=20
DEVICE=None
if RECORD:
 source_path=record_wav(ROOT/'data/raw/recordings',SECONDS,DEVICE,None)
 recorded=load_audio(source_path,Config(max_seconds=180))
 y,sr=recorded.y,recorded.sr
else:
 selected=next(s for s in cfg['series'] if s['series_id']==SERIES_ID)
 y=r.y[int(selected['start_s']*r.sr):int(selected['end_s']*r.sr)];sr=r.sr
print('Nowe nagranie' if RECORD else SERIES_ID)
display(Audio(y,rate=sr,normalize=False))
MANUAL_ONSETS=None
MIN_GAP=.4
SENSITIVITY=.15
DROP_DB=30
QUIET_HOLD=.08
"""),
    md(
        "## Uderzenia i wybrzmienia\nCzerwony: umowne pierwsze 30 ms. Zielony: obserwowany zanik. Pomarańczowy: brak zaniku przed następnym zdarzeniem lub końcem pliku. To heurystyka; szum też wpływa na granicę. Zaznacz myszą fragment wykresu, aby go przybliżyć."
    ),
    code("""segments,t,db=segment_ringdown(y,sr,MANUAL_ONSETS,min_gap_s=MIN_GAP,sensitivity=SENSITIVITY,drop_db=DROP_DB,quiet_hold_s=QUIET_HOLD)
fig=ringdown_plot(y,sr,segments,t,db)
fig.show(renderer='plotly_mimetype')
fig.write_html(OUT/'wybrzmienia.html',include_plotlyjs=True)
display(segments.round(3))
paths=export_pings(y,sr,segments,OUT/'pings')
print('Wyeksportowane fragmenty:',len(paths))
for row in list(segments.itertuples())[:12]:
 print('Ping',row.ping,row.status)
 display(Audio(y[max(0,int((row.onset_s-.015)*sr)):int(row.end_s*sr)],rate=sr,normalize=False))
"""),
]
write(name, body)
print("Built", len(list(DEST.glob("*.ipynb"))), "notebooks")

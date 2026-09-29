"""Build a private HTML atlas and Run All notebook from local client-audit outputs."""

import json
import shutil
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
import soundfile as sf
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/runs/client-20260929"
SOURCE = ROOT / "data/raw/client-20260929/Recordings"


def publish():
    records = pd.read_csv(OUT / "manifest.csv").fillna("")
    events = pd.read_csv(OUT / "candidate_audit.csv")
    good = events[events.accepted]
    (OUT / "audio").mkdir(exist_ok=True)
    atlas = []
    for r in records.itertuples():
        src = SOURCE / r.path
        audio = f"audio/{r.recording_id}{src.suffix}"
        shutil.copy2(src, OUT / audio)
        y, sr = sf.read(OUT / "decoded" / f"{r.recording_id}.wav")
        stride = max(1, len(y) // 5000)
        n = len(y) // stride
        b = y[: n * stride].reshape(n, stride)
        e = events[events.recording_id == r.recording_id]
        atlas.append(
            dict(
                id=r.recording_id,
                path=r.path,
                audio=audio,
                image=f"figures/{r.recording_id}.png",
                t=(np.arange(n) * stride / sr).round(4).tolist(),
                low=b.min(axis=1).round(7).tolist(),
                high=b.max(axis=1).round(7).tolist(),
                accepted=e[e.accepted].onset_s.round(4).tolist(),
                rejected=e[~e.accepted].onset_s.round(4).tolist(),
                duration=round(r.duration_s, 2),
                count=int(r.n_candidates),
                features=int(r.n_feature_candidates),
                gap=round(r.median_gap_s, 3),
                codec=r.codec,
            )
        )
    (OUT / "plotly.min.js").write_text(get_plotlyjs())
    title = "Pierwsze nagrania klienta · 29.09.2026"
    style = """body{font:17px/1.55 system-ui,sans-serif;color:#233640;background:#f7f5f0;margin:0}main{max-width:1080px;margin:auto;padding:40px 28px}h1{font-size:38px;line-height:1.1;max-width:850px}h2{font-size:25px;margin:38px 0 12px}h3{font-size:19px}a{color:#117b80}nav{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}section{border-top:1px solid #d9d8d1;margin-top:30px;padding-top:12px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:white;border:1px solid #dedfd7;border-radius:10px;padding:16px}.card b{display:block;font-size:28px}.note{background:#e6efec;border-left:4px solid #197b77;padding:16px}.warn{background:#f8ead9;border-left:4px solid #b47b3f;padding:16px}img{max-width:100%;background:white}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:8px;border-bottom:1px solid #dadbd5}th{background:#e9eae4}details{margin:15px 0;background:white;padding:14px;border-radius:8px}summary{cursor:pointer;font-weight:650}select{max-width:100%;width:100%;padding:12px;font:inherit}audio{width:100%;margin:12px 0}.scroll{overflow-x:auto}.small{font-size:14px;color:#5b686c}.photos{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.photos img{max-height:260px;object-fit:contain}li{margin:8px 0}code{background:#e9e7df;padding:2px 5px}@media(max-width:700px){.cards,.photos{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}main{padding:22px 15px}}"""
    table = records[
        [
            "recording_id",
            "path",
            "codec",
            "duration_s",
            "n_candidates",
            "n_feature_candidates",
            "median_gap_s",
        ]
    ].rename(
        columns={
            "recording_id": "ID",
            "path": "Plik",
            "duration_s": "s",
            "n_candidates": "Kandydaci",
            "n_feature_candidates": "Do cech",
            "median_gap_s": "Odstęp [s]",
        }
    )
    body = f"""<h1>{title}</h1><p>Ocena przydatności nagrań i plan następnego pomiaru. <strong>Wstępna eksploracja, bez deklaracji skuteczności klasyfikatora.</strong></p>
    <nav><a href="#wnioski">Wnioski</a><a href="#atlas">Odsłuch i segmentacja</a><a href="#cechy">Cechy / AST</a><a href="#noise">Hałas</a><a href="#next">Do klienta</a><a href="analiza.html">Pełny notebook HTML</a></nav>
    <div class="cards"><div class="card"><b>{len(records)}</b>plików audio</div><div class="card"><b>{records.duration_s.sum() / 60:.1f} min</b>łącznie</div><div class="card"><b>2 NOK</b>z pęknięciem według opisu</div><div class="card"><b>{len(good)}</b>kandydatów do cech, nie niezależnych próbek</div></div>
    <section id="wnioski"><h2>Materiał wystarcza do poprawy protokołu</h2>
    <p class="note">Uderzenia i rezonanse są czytelne w przebiegach oraz widmach. Możemy segmentować, porównywać ustawienia i wyznaczać cechy. <strong>Nie mamy jeszcze podstaw do rzetelnego pomiaru accuracy ani recall na nowych płytkach.</strong></p>
    <ol><li><strong>Brakuje identyfikacji OK.</strong> Pięć nagrań OK znajduje się wyłącznie w OLD. Nie wiadomo, ile reprezentują fizycznych płytek ani czy trzy OLD/NOK powtarzają Bad 1–2.</li>
    <li><strong>Ustawienie wyraźnie zmienia brzmienie.</strong> Dla Bad 2 mediana środka widma wynosi 2,18–5,80 kHz między plikami. Podparcie, kierunek, narzędzie i miejsce trafienia nie są rozdzielone w kontrolowanym eksperymencie.</li>
    <li><strong>Tempo jest za szybkie do pełnego ogona.</strong> Typowy odstęp to około 0,26 s. Okno 350 ms zwykle obejmowałoby kolejne zdarzenie. Używamy 150 ms po początku, bez dopisywania ciszy.</li>
    <li><strong>Możliwa ingerencja w tło.</strong> W FACTORY DIRTY NOISE około 20,3% próbek ma dokładnie wartość zero; 22,9% ramek RMS jest poniżej −100 dBFS. Szczególnie podejrzany jest odcinek 11–18 s. To powód do sprawdzenia toru nagrania, nie rozpoznanie konkretnego filtra.</li></ol>
    <p class="small">Wszystkie pliki: 48 kHz, 2 identyczne kanały (dual mono). 19 AAC/M4A i 5 PCM/WAV. Brak próbek ≥0,999 pełnej skali; nie wyklucza to wcześniejszej obróbki sygnału. Zdjęcia obejmują Bad 1–4. „Bitten” pokazujemy osobno jako prawdopodobne wyszczerbienia, do potwierdzenia; nie zwiększamy nimi liczby płyt z pęknięciem.</p></section>
    <section id="atlas"><h2>Odsłuch i kontrola segmentacji</h2><p>Zielone kreski: kandydaci przyjęci do ekstrakcji cech. Szare: odrzuceni przez reguły jakości lub osobny plik tła. Przybliż wykres przeciągnięciem myszą, podwójne kliknięcie przywraca widok. Odsłuch każdego trafienia i zapowiedzi wymaga jeszcze potwierdzenia człowieka.</p>
    <select id="record"></select><p id="meta" class="small"></p><audio id="player" controls preload="none"></audio><div id="wave" style="height:320px"></div><img id="spec" alt="Przebieg i spektrogram wybranego nagrania">
    <details><summary>Wszystkie pliki i liczby kandydatów</summary><div class="scroll">{table.round(3).to_html(index=False, escape=True, border=0)}</div></details>
    <p class="small">Detektor RMS: minimalny odstęp 120 ms. Do cech: co najmniej 12 dB kontrastu ataku wobec poprzednich 80 ms, bez pierwszych 0,75 s i ostatnich 0,4 s, pełne okno 20 ms przed + 150 ms po. To jawne reguły wstępne, nie zatwierdzone etykiety uderzeń. Zmiana odstępu na 250 ms zmniejsza liczbę kandydatów z 2395 do 1789 (łącznie z tłem). W R02 10–12 s widać 8 impulsów, w R06 — 10; zbyt długi odstęp je pomija.</p></section>
    <section id="cechy"><h2>Co zmienia dźwięk?</h2><p>Najpierw porównujemy nagrania tej samej płytki. Wykresy poniżej korzystają z normalizowanych uderzeń: sama różnica głośności nie tłumaczy całej zmiany widma. Dominujący pik może przeskakiwać między rezonansami; nie oznacza to, że materiał zmienił własną częstotliwość o kilka kHz.</p>
    <img src="figures/spectral_conditions.png" alt="Widma Bad 1, Bad 2 i OLD OK">
    <h3>Mapy reprezentacji</h3><p>Każdy punkt to <strong>nagranie</strong>. Kolor: zielony OK, pomarańczowy NOK, fioletowy Bitten. Trójkąty pochodzą z OLD, koła z nowych folderów. PCA dopasowane do całości służy tylko do oglądania; dwie osie nie pokazują całej informacji.</p>
    <div class="note">Na tych mapach nie widać czystego, wspólnego rozdziału OK/NOK odpornego na ustawienie. Nie dowodzi to braku sygnału wady. Różnice między warunkami tej samej płytki są porównywalne z różnicami między płytkami — najpierw trzeba ograniczyć tę zmienność.</div>
    <details open><summary>Physics — 15 prostych cech</summary><img src="figures/pca_physics.png" alt="PCA physics"></details>
    <details><summary>MFCC i mel</summary><img src="figures/pca_mfcc.png" alt="PCA MFCC"><img src="figures/pca_mel.png" alt="PCA mel"></details>
    <details><summary>Zamrożony AST — 115 reprezentatywnych zdarzeń</summary><p>Pięć zdarzeń rozłożonych w czasie z każdego nagrania, średni embedding na plik, bez uczenia sieci ani klasyfikatora. AST pracuje przy 16 kHz, więc nie wykorzystuje pasma powyżej 8 kHz. Ten przegląd nie obejmuje wszystkich uderzeń.</p><img src="figures/pca_ast.png" alt="PCA AST"></details>
    <p>Pliki Bad 2 z opisem „14 to 22” przeanalizowano również według tego przedziału. Dominujący pik pozostaje taki sam wewnątrz i poza nim; zmiana centroidu zależy od nagrania. Bez mapy pozycji nie przypisujemy pojedynczym trafieniom etykiety „nad pęknięciem”.</p>
    <details><summary>Dlaczego bez tabeli accuracy?</summary><p>Losowy podział 2308 uderzeń sprawdzałby rozpoznawanie tych samych płyt i ustawień. Podział po plikach również może przepuścić tę samą sztukę do treningu i testu. Brakuje powiązania OLD z płytkami, a w potwierdzonym zakresie mamy tylko dwie pęknięte sztuki. Nie wybraliśmy progu ani modelu na tych danych. Po uzupełnieniu ID można zrobić mały, jawnie eksploracyjny test z całymi płytkami poza treningiem.</p></details></section>
    <section id="noise"><h2>Hałas i wybrzmienie</h2><p>Test: 115 zdarzeń, dodane realne tło z odcinka 19–30 s, poziomy SNR 20/10/5/0 dB. SNR liczymy z RMS całego okna uderzenia. Mierzymy zmianę cech, <strong>nie trafność modelu</strong>. Wartości porównujemy w obrębie reprezentacji; nie są rankingiem jakości modeli.</p>
    <img src="figures/noise_features.png" alt="Wpływ hałasu na cechy">
    <p>Ściszenie o 12 dB praktycznie nie zmienia cech po normalizacji. Dodanie tła już je zmienia, szczególnie cechy physics przy silnym szumie. Nie dowodzi to odporności MFCC; rzeczywisty hałas, odbicia i ustawienie trzeba sprawdzić w następnej sesji.</p>
    <p class="warn">Otrzymaliśmy tylko 31,6 s tła, z fragmentami niemal ciszy. Nie używamy tego nagrania do kalibracji poziomu hałasu fabryki ani do wiarygodnego estymowania tłumienia płyt. Typowy opis zaniku o 10 dB z tej paczki wynosi 20–40 ms, lecz może obejmować skutki toru nagrania.</p></section>
    <section id="next"><h2>Uwagi do następnej sesji</h2><ol>
    <li><strong>Jedno ID na fizyczną płytkę.</strong> Uzupełnić powiązania OLD, potwierdzić status Bitten, dodać typ/wymiary, temperaturę i sposób potwierdzenia wady. Zdjęcie z tym samym ID i zaznaczonym pęknięciem.</li>
    <li><strong>Krótki test toru nagrywania.</strong> Bez odszumiania, bramki, AGC, kompresji i trybu „mowa”. Bezpośrednio do WAV PCM 48 kHz, 24 lub 16 bit. Nie konwertować starych AAC „dla poprawy jakości”. Zrobić 10 s tła → kilka uderzeń → 10 s tła i sprawdzić, czy tło nie znika.</li>
    <li><strong>Jedno wspólne ustawienie dla OK i NOK.</strong> To samo narzędzie, podparcie, mikrofon, dystans i gain. Zmiany testować parami na obu klasach i zapisywać jako osobne warianty. Nie wybierać zwycięskiego ustawienia wyłącznie z tych dwóch NOK.</li>
    <li><strong>10–15 pojedynczych uderzeń na płytkę.</strong> Na początek odstępy 1–2 s, bez dobijania i rozmowy. Numerowana siatka tych samych miejsc dla każdej sztuki. Dodatkowe trafienia blisko pęknięcia jako osobny blok z czasem i pozycją.</li>
    <li><strong>Więcej niezależnych OK.</strong> W najbliższej partii choć kilka różnych OK zmierzonych identycznie jak dostępne NOK. Zbierać kolejne różne pęknięte płytki, gdy tylko się pojawią. Setki trafień w dwie sztuki nie zwiększają różnorodności wad.</li>
    <li><strong>Powtórzyć ustawienie i tło.</strong> Zdjąć oraz ponownie ułożyć każdą płytkę, powtórzyć serię; nagrywać OK/NOK naprzemiennie. Dostarczyć kilka minut ciągłego tła z tych samych ustawień, przy różnych stanach maszyn; docelowo około 10 min.</li></ol>
    <p><a href="UWAGI_DO_KLIENTA.md">Krótki tekst do przekazania klientowi</a> · <a href="metadata_do_uzupelnienia.csv">Tabela metadanych</a> · <a href="features.csv">Cechy CSV</a> · <a href="candidate_audit.csv">Audyt kandydatów CSV</a></p></section>
    <p class="small">Analiza lokalna. Bez wysyłania danych do usług zewnętrznych. Etykiety na podstawie nazw plików/folderów oraz opisu użytkownika; fotografie nie zastępują potwierdzenia technicznego. Przegląd sygnałów wykonano programowo i wizualnie, pełny odsłuch człowieka pozostaje do zatwierdzenia.</p>"""
    js = """const recordings=DATA; const select=document.getElementById('record');recordings.forEach(r=>{const o=document.createElement('option');o.value=r.id;o.textContent=r.id+' · '+r.path;select.appendChild(o)});function show(){const r=recordings.find(r=>r.id===select.value);document.getElementById('meta').textContent=r.duration+' s · '+r.codec+' · kandydaci: '+r.count+' · do cech: '+r.features+' · mediana odstępu: '+r.gap+' s';document.getElementById('player').src=r.audio;document.getElementById('spec').src=r.image;const shapes=[...r.accepted.map(t=>({type:'line',x0:t,x1:t,y0:0,y1:1,yref:'paper',line:{color:'#137f83',width:1}})),...r.rejected.map(t=>({type:'line',x0:t,x1:t,y0:0,y1:1,yref:'paper',line:{color:'#bda48c',width:1,dash:'dot'}}))];Plotly.react('wave',[{x:r.t,y:r.low,type:'scatter',mode:'lines',line:{color:'#4b6875',width:1},showlegend:false},{x:r.t,y:r.high,type:'scatter',mode:'lines',fill:'tonexty',line:{color:'#4b6875',width:1},showlegend:false}],{margin:{l:65,r:20,t:15,b:45},xaxis:{title:{text:'Czas [s]'}},yaxis:{title:{text:'Amplituda'}},shapes,dragmode:'zoom'},{responsive:true,displaylogo:false})}select.onchange=show;select.value='R06';show();""".replace(
        "DATA", json.dumps(atlas, ensure_ascii=False).replace("</", "<\\/")
    )
    (OUT / "START.html").write_text(
        '<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'
        + title
        + "</title><style>"
        + style
        + '</style><script src="plotly.min.js"></script><main>'
        + body
        + "</main><script>"
        + js
        + "</script></html>"
    )
    meta = records[["recording_id", "path", "plate_id", "label", "scope"]].copy()
    for c in [
        "confirmed_plate_id",
        "same_as_other_recording",
        "confirmed_defect_type",
        "plate_type",
        "dimensions",
        "temperature",
        "support",
        "tool",
        "mic_distance",
        "gain",
        "processing_enabled",
        "tap_positions",
        "session_id",
        "label_confirmation",
    ]:
        meta[c] = ""
    if not (OUT / "metadata_do_uzupelnienia.csv").exists():
        meta.to_csv(OUT / "metadata_do_uzupelnienia.csv", index=False)


def notebook():
    folder = ROOT / "notebooks/client"
    folder.mkdir(exist_ok=True)
    cells = []

    def md(x):
        cells.append(nbf.v4.new_markdown_cell(x))

    def code(x):
        cells.append(nbf.v4.new_code_cell(x.strip()))

    md("""# Pierwsza paczka klienta — jakość, uderzenia i porównanie nagrań
**Run All** przelicza lokalne nagrania narzędziami `pinglab`, generuje wykresy, CSV i raport `START.html`. Wybierz kernel `.venv` / Ping Lab. Dane są w `data/raw/client-20260929/Recordings` (prywatna kopia ZIP; oryginały bez zmian).

Bad 1–2 traktujemy roboczo jako dwie pęknięte płytki. Bitten analizujemy osobno. ID płytek w OLD są nieznane. Nie dzielimy losowo uderzeń na trening i test i nie raportujemy accuracy bez poprawnego podziału po fizycznych płytkach.""")
    code("""from pathlib import Path
import sys
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/pinglab').is_dir())
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from IPython.display import display, Markdown, Image, Audio
import pandas as pd
import numpy as np
import soundfile as sf
from analyze_client_recordings import run, OUT
SOURCE=ROOT/'data/raw/client-20260929/Recordings'
RUN_AST=True # lokalne wagi i cache, bez pobierania; ustaw False dla szybszego przeglądu
records, events=run(SOURCE)
print(f'{len(records)} pliki | {records.duration_s.sum()/60:.1f} min | {len(events)} kandydatów do cech')""")
    md("""## 1. Co faktycznie dostaliśmy?
Jednostką oceny jest fizyczna płytka. Licznik zdarzeń nie oznacza liczby niezależnych przykładów. Kolumna `plate_id` dla Bad pochodzi z folderu i wymaga potwierdzenia; dla OLD pozostaje pusta.""")
    code("""display(records.groupby(['folder','label']).agg(pliki=('recording_id','size'),sekundy=('duration_s','sum'),kandydaci_do_cech=('n_feature_candidates','sum')).round(1))
display(records[['recording_id','path','native_sr','codec','duration_s','peak_dbfs','clipping_fraction','zero_fraction','channel_max_difference']].round(4))""")
    md("""## 2. Nagranie pod lupą — zmień RECORDING_ID
Wszystkie pliki są 48 kHz; kanały stereo są identyczne. Brak clippingu pełnej skali nie wyklucza wcześniejszej kompresji/odszumiania. W tło (`R16`) wpisane są długie odcinki niemal ciszy. Odsłuch jest oryginalnej głośności, bez automatycznego wyrównania.""")
    code("""RECORDING_ID='R06'
r=records.set_index('recording_id').loc[RECORDING_ID]
print(r.path)
y,sr=sf.read(OUT/'decoded'/f'{RECORDING_ID}.wav')
display(Audio(y,rate=sr,normalize=False));display(Image(filename=str(OUT/'figures'/f'{RECORDING_ID}.png')))
display(records[['recording_id','quiet_intervals']].query("recording_id in ['R16','R23','R24']"))""")
    md("""## 3. Segmentacja i tempo
Minimalny odstęp 250 ms z poprzedniego przykładu pomija szybkie trafienia. Tu używamy 120 ms. Reguły wstępne: kontrast ataku ≥12 dB, bez brzegów nagrania, pełne okno 20 ms przed i 150 ms po uderzeniu. Nie wymuszamy liczby trafień. Zdarzenia należy zatwierdzić odsłuchem — detektor wykrywa także inne impulsy.

Wykres jest klikalny: zaznacz fragment myszą, aby go przybliżyć. Kolory znaczą przyjęcie do cech, nie klasę OK/NOK.""")
    code("""import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='plotly_mimetype'
audit=pd.read_csv(OUT/'candidate_audit.csv'); e=audit[audit.recording_id==RECORDING_ID]
step=max(1,len(y)//6000);n=len(y)//step;b=y[:n*step].reshape(n,step);t=np.arange(n)*step/sr
fig=go.Figure([go.Scatter(x=t,y=b.min(1),mode='lines',line_color='#476675',showlegend=False),go.Scatter(x=t,y=b.max(1),mode='lines',line_color='#476675',showlegend=False)])
for hit in e.itertuples():fig.add_vline(x=hit.onset_s,line_color='#137f83' if hit.accepted else '#bda48c',line_width=1)
fig.update_layout(title=RECORDING_ID+' — kandydaci ataków',xaxis_title='Czas [s]',yaxis_title='Amplituda',height=350);fig.show()
display(pd.read_csv(OUT/'detector_sensitivity.csv').pivot(index='recording_id',columns='min_gap_s',values='n_candidates'))
display(e[['event_id','onset_s','attack_contrast_db','accepted','rejection']].head(12).round(3))""")
    md("""## 4. Dwa trafienia obok siebie
Zmień numery w EVENT_A/B. To porównanie kształtu widma i odsłuchu, bez nadawania trafieniu pewnej etykiety wady. Wybrane domyślnie trafienia są blisko środka czasowego danego nagrania.""")
    code("""from pinglab.features import extract_features
from pinglab.plots import compare_hits
EVENT_A=events[events.recording_id=='R06'].iloc[len(events[events.recording_id=='R06'])//2].event_id
EVENT_B=events[events.recording_id=='R23'].iloc[len(events[events.recording_id=='R23'])//2].event_id
ya,sr=sf.read(OUT/'clips'/f'{EVENT_A}.wav');yb,_=sf.read(OUT/'clips'/f'{EVENT_B}.wav')
print(EVENT_A,EVENT_B)
display(Audio(ya,rate=sr,normalize=True));display(Audio(yb,rate=sr,normalize=True))
compare_hits([ya,yb],sr=sr,names=[EVENT_A,EVENT_B],offsets=[.02,.02]).show()
""")
    md("""## 5. Widma, cechy i zmienność ustawienia
W każdej mapie punkt oznacza medianę cech nagrania; AST uśrednia 5 embeddingów. Zielony = OK, pomarańczowy = NOK, fioletowy = Bitten. Trójkąt = OLD, koło = pozostałe. PCA dopasowane do całej paczki nie mierzy skuteczności.

Physics: rezonans, środek i szerokość widma, roll-off, płaskość, entropia, zanik o 10 dB, energia późna/wczesna, liczba pików i sześć pasm. MFCC/mel opisują barwę. AST jest zamrożonym modelem AudioSet (115 zdarzeń, nie wszystkie); nie uczymy go na tej paczce.""")
    code("""from client_recording_diagnostics import run as run_diagnostics
noise_summary=run_diagnostics(run_ast=RUN_AST)
display(Image(filename=str(OUT/'figures/spectral_conditions.png')))
for family in ['physics','mfcc','mel'] + (['ast'] if RUN_AST else []):
 display(Image(filename=str(OUT/'figures'/f'pca_{family}.png')))
display(pd.read_csv(OUT/'recording_summary.csv')[['recording_id','path','peak_hz','centroid_hz','decay10_s']].round(2))
display(pd.read_csv(OUT/'near_crack.csv').round(2))""")
    md("""## 6. Rzeczywiste tło: test wrażliwości cech
Dodajemy fragmenty tła z 19–30 s pliku R16, omijając długą ciszę. SNR dotyczy RMS całego okna. Test 115 zdarzeń pokazuje zmianę reprezentacji, nie accuracy. Skale physics/MFCC/mel różnią się, więc niższy słupek nie oznacza automatycznie lepszego klasyfikatora. Samo ściszenie o 12 dB jest kontrolą normalizacji.

Tło jest krótkie i prawdopodobnie przetworzone; ta symulacja nie zastępuje nagrań przy rzeczywistym hałasie stanowiska.""")
    code("""display(noise_summary.round(3))
display(Image(filename=str(OUT/'figures/noise_features.png')))
yn,sr=sf.read(OUT/'decoded/R16.wav');display(Audio(yn,rate=sr,normalize=False))""")
    md("""## 7. Co możemy powiedzieć klientowi?
**Sygnał nadaje się do eksploracji. Najpilniejsze są porównywalne pomiary i metadane.**

- Uzupełnić ID fizycznych płytek w OLD i potwierdzić związek z Bad 1–2; Bitten trzymać osobno.
- Sprawdzić, czy nagrywanie usuwa tło: wyłączyć AGC/odszumianie/bramkę, nagrać próbę do WAV.
- Zachować identyczne podparcie, narzędzie, mikrofon i gain dla obu klas.
- 10–15 pojedynczych trafień w te same ponumerowane miejsca, co 1–2 s; osobny blok blisko pęknięcia.
- Dołożyć różne płytki OK, kolejne NOK w miarę dostępności, serię po ponownym ułożeniu i ciągły szum stanowiska.

Nie raportujemy accuracy/recall ani optymalnego progu. Nie znamy niezależności OK, a dwie pęknięte sztuki nie reprezentują różnorodności wad. Wizualne nakładanie się punktów nie dowodzi, że zadanie jest niemożliwe.""")
    code("""from build_client_review import publish
publish()
assert not events.label.eq('background').any()
assert events.duration_s.between(.169,.171).all(), 'Cechy muszą mieć jednakowe okno 170 ms'
assert records.sha256.nunique()==len(records)
assert events.event_id.is_unique
from pinglab.features import feature_columns
assert np.isfinite(events[feature_columns(events,'physics')+feature_columns(events,'mfcc')+feature_columns(events,'mel')]).all().all()
print('Raport:',OUT/'START.html')
print('Kontrole: okna równe, brak tła w cechach, unikalne źródła i zdarzenia, wartości skończone.')""")
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = dict(name="pinglab", display_name="Python (pinglab)", language="python")
    nbf.write(nb, folder / "01_wstepna_analiza.ipynb")


if __name__ == "__main__":
    publish()
    notebook()

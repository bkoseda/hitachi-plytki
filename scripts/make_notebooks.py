from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BOOT = """from pathlib import Path
import os, json, sys
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
assert (ROOT / "pyproject.toml").exists(), "Otwórz cały folder projektu w VS Code."
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
from pinglab.audio import Config, load_audio, detect_hits, cut_hits, listen
from pinglab.session import load_session
SESSION = load_session(ROOT)
CONFIG, FMAX = SESSION.config, SESSION.fmax
MANIFEST, ANNOTATIONS = SESSION.manifest, SESSION.annotations
print("Dane:", SESSION.source_kind, "· manifest:", MANIFEST.relative_to(ROOT) if MANIFEST.is_relative_to(ROOT) else MANIFEST.name)
print("Wspólne pasmo do", FMAX, "Hz · konfiguracja: project.json")
if SESSION.source_kind == "synthetic":
    display(Markdown("**SYMULACJA — wynik sprawdza narzędzia, nie wykrywanie pęknięć u klienta.**"))
"""


def write(name, cells):
    nb = nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in cells]
    )
    nb.metadata = dict(
        kernelspec=dict(display_name="Ping Lab (Python)", language="python", name="pinglab"),
        language_info=dict(name="python", version="3.12"),
    )
    destination = ROOT / "notebooks" / name
    if destination.exists():
        old = nbf.read(destination, as_version=4)
        if [(c.cell_type, c.source) for c in old.cells] == [(c.cell_type, c.source) for c in nb.cells]:
            return
    nbf.write(nb, destination)


write(
    "01_pierwsza_para.ipynb",
    [
        (
            "md",
            "# 01 · Pierwsze dwa pliki\n**Cel:** odsłuchać, sprawdzić jakość i zobaczyć różnice. Około 10 minut.\n\nDomyślnie otwierasz **symulację**, nie pomiary klienta. Wspólne dane ustaw w `project.json`. Pracujemy lokalnie. `Shift+Enter` uruchamia komórkę; **Run → Run All Cells** uruchamia całość.",
        ),
        ("code", BOOT),
        (
            "md",
            "## Wrzuć pliki\nOpcjonalny przycisk zapisze je do `data/raw/inbox/`. Możesz też skopiować je Finderem. Dodaj pliki do manifestu wskazanego w `project.json`.",
        ),
        ("code", 'SHOW_UPLOAD_WIDGET = False  # opcjonalne; VS Code może poprosić o pobranie zasobów widgetów\nif SHOW_UPLOAD_WIDGET:\n    from pinglab.widgets import upload_box\n    uploader = upload_box(ROOT / "data/raw/inbox")'),
        (
            "code",
            """# KONFIGURACJA — jedyne obowiązkowe zmiany po otrzymaniu plików
REC_A, REC_B = SESSION.pair()
FILE_A, FILE_B = Path(REC_A.resolved_path), Path(REC_B.resolved_path)
NAME_A, NAME_B = f"A · {REC_A.plate_id} ({REC_A.label})", f"B · {REC_B.plate_id} ({REC_B.label})"
CHANNEL_A = int(REC_A.channel) if REC_A.channel else None
CHANNEL_B = int(REC_B.channel) if REC_B.channel else None  # None = kanał z najwyższym RMS; 0/1 wymusza kanał
saved_times = json.loads(ANNOTATIONS.read_text()) if ANNOTATIONS.exists() else {}
MANUAL_A, MANUAL_B = saved_times.get(REC_A.path), saved_times.get(REC_B.path)  # np. [0.4, 1.9, 3.4]; [] = brak uderzeń
HIT_A, HIT_B = 0, 0             # numer uderzenia od zera
""",
        ),
        (
            "code",
            """a = load_audio(FILE_A, CONFIG, channel=CHANNEL_A)
b = load_audio(FILE_B, CONFIG, channel=CHANNEL_B)
display(pd.DataFrame([a.qc,b.qc], index=[NAME_A,NAME_B]).T)
for name, audio in [(NAME_A,a),(NAME_B,b)]:
    print(name, "· głośność oryginalna")
    display(listen(audio.y,audio.sr,normalize=False))""",
        ),
        (
            "md",
            "**Co sprawdzić:** przesterowanie, mowa podczas stuknięcia, dźwięk dotknięcia mikrofonu, za krótki ogon. `dBFS` mierzy poziom cyfrowy, nie głośność w fabryce. Mała amplituda nie oznacza automatycznie złej płytki.",
        ),
        (
            "code",
            """from pinglab.plots import inspection, compare_hits
onsets_a = detect_hits(a.y,a.sr,CONFIG) if MANUAL_A is None else np.array(MANUAL_A)
onsets_b = detect_hits(b.y,b.sr,CONFIG) if MANUAL_B is None else np.array(MANUAL_B)
print("A:", onsets_a, "B:", onsets_b)
display(inspection(a,onsets_a))
display(inspection(b,onsets_b))""",
        ),
        (
            "md",
            "## Porównaj podobne miejsca uderzenia\nJeśli automat podzielił źle: wpisz poprawne sekundy do `MANUAL_A/B` i uruchom ponownie. Nie traktuj każdego głośnego zdarzenia jak uderzenia w ceramikę. Poniższy odsłuch wyrównuje szczyty: służy do porównania barwy, nie głośności.",
        ),
        (
            "code",
            """clips_a = cut_hits(a.y,a.sr,onsets_a,CONFIG)
clips_b = cut_hits(b.y,b.sr,onsets_b,CONFIG)
assert clips_a and clips_b, "Brak uderzeń. Uzupełnij MANUAL_A / MANUAL_B."
ca,cb = clips_a[HIT_A],clips_b[HIT_B]
for name,c in [(NAME_A,ca),(NAME_B,cb)]:
    print(name, "· wyrównany szczyt", "· skrócone okno" if c['truncated'] else "")
    display(listen(c['y'],CONFIG.sr,normalize=True))
display(compare_hits([ca['y'],cb['y']],CONFIG.sr,[NAME_A,NAME_B],
                     [ca['onset_offset_s'],cb['onset_offset_s']],FMAX))""",
        ),
        (
            "code",
            """from pinglab.features import extract_features
assert FMAX <= min(a.native_sr,b.native_sr)/2, "Obniż FMAX do wspólnego pasma źródeł."
fa = extract_features(ca['y'],CONFIG.sr,ca['onset_offset_s'],FMAX)
fb = extract_features(cb['y'],CONFIG.sr,cb['onset_offset_s'],FMAX)
keys=['peak_hz','centroid_hz','bandwidth_hz','decay10_s','late_early_ratio','flatness']
display(pd.DataFrame({NAME_A:fa,NAME_B:fb}).loc[keys].round(3))""",
        ),
        (
            "md",
            "## Feedback dla zespołu pomiarowego\nUzupełnij trzy zdania: **Jakość:** … **Różnica A/B:** … **Następne nagranie:** …\n\nPrzesunięty rezonans lub krótsze wybrzmiewanie to trop. Mogą też wynikać z podparcia, geometrii, miejsca uderzenia albo mikrofonu. Z dwóch plików nie licz skuteczności. Dalej: **02** — cała seria i korekta podziału.",
        ),
    ],
)

write(
    "02_seria_i_uderzenia.ipynb",
    [
        (
            "md",
            "# 02 · Seria i uderzenia\n**Cel:** utrzymać porządek w danych, sprawdzić każdy podział i wyeksportować cechy. Jeden `plate_id` oznacza jeden fizyczny detal — także gdy nagrasz go w kilku plikach.",
        ),
        ("code", BOOT),
        (
            "code",
            """# KONFIGURACJA
MANIFEST = SESSION.manifest
ANNOTATIONS = SESSION.annotations
OUT = SESSION.output / "batch"
""",
        ),
        (
            "md",
            "## Manifest\nWiersz = plik. Obowiązkowe: `path`, `plate_id`, `label`, `session_id`. Etykiety: `ok`, `nok`, `unknown`, `background`. Ścieżka względna jest liczona **od pliku CSV**. Szablon: `data/manifest.template.csv`.\n\n`channel` = opcjonalny numer kanału (0/1, puste = auto). `position` = miejsce uderzenia; `environment` = warunki; `notes` = opis wady/podparcia; `photo` = ścieżka zdjęcia; `plate_type` = typ płyty; `temperature` = temperatura. Przy wielu miejscach w jednym pliku wpisz kolejność i zachowuj ten sam protokół.",
        ),
        (
            "code",
            """from pinglab.dataset import read_manifest, analyze_manifest
manifest = read_manifest(MANIFEST)
display(manifest.drop(columns=['resolved_path','sha256']).head(12))
print("Fizyczne płytki:",manifest[manifest.label!='background'].groupby('label').plate_id.nunique().to_dict())""",
        ),
        (
            "md",
            "## Zatwierdź uderzenia\nWpisz poprawione sekundy do `CORRECTIONS` i uruchom ponownie. Puste `{}` niczego nie zmienia. Opcjonalnie ustaw `SHOW_REVIEW_WIDGET=True`, aby użyć przycisków edytora (VS Code może poprosić o pobranie zasobów). Większy próg wykrywa mniej kandydatów. Zatwierdzone czasy mają pierwszeństwo przed automatem; aby przeliczyć je automatycznie, usuń wpis z JSON. Puste pole i zapis oznaczają świadome odrzucenie wszystkich uderzeń.",
        ),
        (
            "code",
            """# Ręczna korekta bez widgetów: klucz = dokładne path z manifestu, wartości = sekundy.
CORRECTIONS = {}  # np. {"sample_a.wav": [0.42, 1.93, 3.48]}; [] odrzuca plik
if CORRECTIONS:
    from pinglab.audio import cut_hits
    known = set(manifest.path)
    assert set(CORRECTIONS) <= known, "Klucz korekty musi zgadzać się z path w manifeście."
    for path, seconds in CORRECTIONS.items():
        rec = manifest[manifest.path == path].iloc[0]
        recording = load_audio(rec.resolved_path,CONFIG,channel=int(rec.channel) if rec.channel else None)
        cut_hits(recording.y,recording.sr,np.asarray(seconds,float),CONFIG)  # walidacja czasów
    existing = json.loads(ANNOTATIONS.read_text()) if ANNOTATIONS.exists() else {}
    ANNOTATIONS.parent.mkdir(parents=True,exist_ok=True)
    ANNOTATIONS.write_text(json.dumps({**existing,**CORRECTIONS},indent=2))
    print("Zapisano korekty:",list(CORRECTIONS))
SHOW_REVIEW_WIDGET = False  # opcjonalne; podstawowy Run All nie wymaga widgetów
if SHOW_REVIEW_WIDGET:
    from pinglab.widgets import review_manifest
    review = review_manifest(MANIFEST,ANNOTATIONS,CONFIG)""",
        ),
        (
            "md",
            "## Przelicz serię\nKomórka działa także przed ręcznym przeglądem, ale wynik zaznaczy niezatwierdzone pliki. Przy zmianie okna / pasma przelicz wszystkie pliki. W notebooku ten katalog wyników jest aktualizowany; do zachowania wersji zmień `OUT`.",
        ),
        (
            "code",
            """events, quality = analyze_manifest(MANIFEST,OUT,CONFIG,FMAX,
                                   ANNOTATIONS if ANNOTATIONS.exists() else None)
display(quality[['path','label','peak_dbfs','clipping_fraction','n_hits','reviewed','warnings']])
print("Uderzenia:", len(events), "· Wyniki:", OUT)
assert not events.empty, "Nie wykryto uderzeń; popraw segmentację."
display(events[['plate_id','label','hit_id','start_s','end_s','truncated']].head())""",
        ),
        (
            "code",
            """from pinglab.report import write_report
source_kind = SESSION.source_kind
report = write_report(events,quality,OUT/'report.html',source_kind=source_kind)
display(Markdown(f"Raport HTML: `{report}`"))""",
        ),
        (
            "md",
            "**Pliki wynikowe:** `quality.csv` — jakość; `features.csv` — cechy każdego uderzenia; `run.json` — parametry i sumy kontrolne. Dalej: **03** — podobieństwa i mały model.",
        ),
    ],
)

write(
    "03_cechy_i_modele.ipynb",
    [
        (
            "md",
            "# 03 · Cechy, prosty klasyfikator, dobre wzorce\n**Cel:** sprawdzić, czy różnice powtarzają się na innych płytkach. Domyślnie używamy mediany cech uderzeń jednej płytki. To stabilna baza, ale może ukryć wadę słyszalną tylko w jednym miejscu — wtedy porównaj osobno stałe miejsca uderzeń.",
        ),
        ("code", BOOT),
        (
            "code",
            """# KONFIGURACJA — użyj tego samego manifestu i parametrów co w 02
MANIFEST = SESSION.manifest
ANNOTATIONS = SESSION.annotations  # dla klienta: zatwierdzony plik z 02
OUT = SESSION.output / "models"
FAMILY = 'resonance'  # resonance (3 cechy) / physics (15) / mfcc / mel
from pinglab.dataset import analyze_manifest
events, quality = analyze_manifest(MANIFEST,OUT,CONFIG,FMAX,
                                    ANNOTATIONS if ANNOTATIONS.exists() else None)""",
        ),
        (
            "md",
            "## Co oznaczają cechy?\n**resonance:** tylko dominujący pik, centroid i czas zaniku — zacznij od małej liczby cech. **physics:** szerszy zestaw 15 cech, w tym pasma i płaskość. **mfcc:** zwięzły opis barwy. **mel:** kształt widma w 40 pasmach.\n\n`decay10_s` to czas zejścia obwiedni o 10 dB w wybranym oknie, nie laboratoryjny pomiar tłumienia. Zmienny hałas i ucięty ogon mogą go zepsuć.",
        ),
        (
            "code",
            """from pinglab.plots import embedding_map,distance_map
from pinglab.learning import plate_table,evaluate,anomaly_table,permutation_check
table,columns = plate_table(events,FAMILY)
display(table[['plate_id','label','n_hits','session_id']].head())
if len(table)>=3:
    display(embedding_map(events,FAMILY)[0])
    display(distance_map(events,FAMILY))""",
        ),
        (
            "md",
            "PCA ściska wiele cech do dwóch osi. Bliskie punkty mają podobne cechy, ale „dwie chmury” nie dowodzą wykrywania defektów. Zmień kolor na sesję, by zobaczyć możliwy wpływ warunków.",
        ),
        (
            "code",
            """if len(table)>=3:
    display(embedding_map(events,FAMILY,color='session_id')[0])""",
        ),
        (
            "md",
            "## Ocena klasyfikatora\nRegresja logistyczna z silną regularyzacją. Każda płytka dostaje prognozę z modelu, który jej nie widział; skalowanie też uczy się tylko na treningu. Minimum programowe: 3 OK + 3 NOK; to nadal bardzo mały test. Przy mniejszej liczbie pokażemy powód zamiast sztucznej skuteczności.",
        ),
        (
            "code",
            """assessment = None
try:
    assessment = evaluate(events,FAMILY)
    metrics,predictions,folds = assessment
    display(pd.Series({k:v for k,v in metrics.items() if k not in ['warning','interval_note']}))
    from pinglab.plots import prediction_plot
    display(prediction_plot(predictions))
    display(predictions)
except ValueError as error:
    print(error)""",
        ),
        (
            "md",
            "**Recall NOK:** ile wadliwych wykryto. **Fałszywe alarmy:** ile dobrych odrzucono. Balanced accuracy uśrednia trafność obu klas. Score 0.7 nie oznacza „70% pewności wady”. Mała próba daje duży błąd oceny. Porównanie wielu metod na tych samych danych jest eksploracją; zwycięzcę sprawdź później na nowych płytkach.",
        ),
        (
            "code",
            """# Ostrzejsza próba: cała nowa sesja poza treningiem.
try:
    session_metrics, session_predictions, session_folds = evaluate(events,FAMILY,by_session=True,bootstrap=0)
    display(pd.Series(session_metrics))
except ValueError as error:
    print("Test sesji:",error)""",
        ),
        (
            "md",
            "## Mam tylko dobre wzorce\nOdległość do najbliższej dobrej płytki pomaga wskazać nietypowe nagrania. Nietypowy nie znaczy uszkodzony. Próg jest roboczy, wyliczony z niewielu dobrych wzorców; nie gwarantuje częstości fałszywych alarmów.",
        ),
        (
            "code",
            """try:
    anomaly, normal_model = anomaly_table(events,FAMILY)
    display(anomaly.sort_values('distance',ascending=False))
except ValueError as error:
    print(error)""",
        ),
        (
            "code",
            """# Opcjonalnie: czy podobny wynik powstaje po przetasowaniu etykiet całych płytek?
RUN_PERMUTATION = False
if RUN_PERMUTATION:
    display(permutation_check(events,FAMILY,repeats=99))""",
        ),
        (
            "code",
            """from pinglab.report import write_report
source_kind=SESSION.source_kind
print(write_report(events,quality,OUT/'report.html',assessment,source_kind=source_kind))""",
        ),
        (
            "md",
            "## Zapis modelu i nowe płytki\nTo opcjonalny krok po sprawdzeniu oceny. Wybierz nowy katalog wersji. Model stosuje ten sam wybór cech i medianę uderzeń. Otwieraj wyłącznie własne pliki `joblib`.",
        ),
        (
            "code",
            """from dataclasses import asdict
from pinglab.model import train_final,predict_new
SAVE_MODEL = False
MODEL_FOLDER = ROOT / "models/my-baseline-v1"
PROCESSING = {'config':asdict(CONFIG),'fmax':FMAX}
if SAVE_MODEL:
    print(train_final(events,MODEL_FOLDER,FAMILY,PROCESSING))
# Nowe dane (z nowymi fizycznymi płytkami) przetwórz analyze_manifest z tym samym CONFIG/FMAX:
# new_events, new_qc = analyze_manifest("data/new/manifest.csv", "reports/runs/new", CONFIG, FMAX)
# display(predict_new(new_events,MODEL_FOLDER,PROCESSING))""",
        ),
        (
            "md",
            "Dalej: **04** — gotowy model jako alternatywna reprezentacja. **05** — kontrola wpływu szumu i sposobu nagrania.",
        ),
    ],
)

write(
    "04_gotowy_model_audio.ipynb",
    [
        (
            "md",
            "# 04 · Gotowy model audio, bez treningu sieci\n**Cel:** sprawdzić alternatywny opis dźwięku. AST zamienia uderzenie na 768 liczb. Używamy wag AudioSet w trybie zamrożonym. To model ogólnych zdarzeń audio, nie wyspecjalizowany detektor pęknięć.\n\nModel wymaga jednorazowego pobrania przez `scripts/warm_model.py`. Ten notebook jest opcjonalny (priorytet 3). Obliczenia pozostają lokalne. CPU jest domyślny; 48 GB RAM wystarcza z dużym zapasem dla pojedynczych wywołań.",
        ),
        ("code", BOOT),
        (
            "code",
            """# KONFIGURACJA
MANIFEST = SESSION.manifest
ANNOTATIONS = SESSION.annotations
OUT = SESSION.output / "ast"
OFFLINE = True  # przy nowej instalacji: najpierw python scripts/warm_model.py
from pinglab.dataset import analyze_manifest
events,quality = analyze_manifest(MANIFEST,OUT,CONFIG,FMAX,ANNOTATIONS if ANNOTATIONS.exists() else None)""",
        ),
        (
            "md",
            "## Oblicz embeddingi\nPierwszy przebieg liczy każde uderzenie; kolejne korzystają z cache zależnego od zawartości audio i wersji modelu. AST widzi sygnał przeskalowany do 16 kHz, więc nie wykorzystuje rezonansów powyżej 8 kHz. Krótkie okna są dopełniane zgodnie z ekstraktorem modelu; to niedopasowanie do treningowych klipów warto sprawdzić.",
        ),
        (
            "code",
            """from pinglab.embeddings import ASTEncoder
import time
start = time.perf_counter()
encoder = ASTEncoder(ROOT/'models/ast',offline=OFFLINE,device='cpu')
encoded = encoder.add_to_events(events,ROOT/'data/cache/ast',CONFIG)
encoded.to_csv(OUT/'features_ast.csv',index=False)
print("Model:",encoder.revision,"· uderzeń:",len(encoded),"· sekund:",round(time.perf_counter()-start,1))""",
        ),
        (
            "code",
            """from pinglab.plots import embedding_map
if encoded.plate_id.nunique()>=3:
    display(embedding_map(encoded,'ast')[0])
    display(embedding_map(encoded,'ast',color='session_id')[0])""",
        ),
        (
            "md",
            "## Ten sam test, inna reprezentacja\nModel AST pozostaje zamrożony. Uczymy wyłącznie mały klasyfikator na embeddingach. Normalizacja i klasyfikator są wewnątrz każdej iteracji testu; podział płytek jest taki sam jak w 03.",
        ),
        (
            "code",
            """from pinglab.learning import evaluate
comparisons=[]
for family in ['resonance','physics','mfcc','ast']:
    try:
        metrics,pred,folds=evaluate(encoded,family,bootstrap=300)
        comparisons.append({k:metrics[k] for k in ['family','n_plates','balanced_accuracy','recall_nok','false_alarm_rate']})
        pred.to_csv(OUT/f'predictions_{family}.csv',index=False)
    except ValueError as error:
        print(family,error)
display(pd.DataFrame(comparisons))""",
        ),
        (
            "md",
            "**Interpretacja:** jeśli klasyczne cechy są równie dobre, zostań przy prostszej metodzie. Jeśli AST grupuje głównie szum / sesję, nie ma powodu mu ufać. Wynik na symulacji nie wybiera metody dla klienta.\n\n**Plan awaryjny:** zamrożone AST / PANNs / CLAP + mały klasyfikator; dopiero przy większej liczbie niezależnych detali rozważ odmrożenie końcowych warstw. Nie trenuj całej sieci na 30 płytkach. Źródła i ograniczenia: `docs/RESEARCH.md`.",
        ),
    ],
)

write(
    "05_odpornosc_i_protokol.ipynb",
    [
        (
            "md",
            "# 05 · Co psuje porównanie?\n**Cel:** zobaczyć wpływ głośności i tła, zanim uznamy różnicę za defekt. Zmodyfikowane wersje jednej płytki to nadal jedna płytka.",
        ),
        ("code", BOOT),
        (
            "code",
            """# KONFIGURACJA
FILE = Path(SESSION.pair()[0].resolved_path)
BACKGROUND = SESSION.background  # np. ROOT / "data/raw/factory_background.wav"
audio=load_audio(FILE,CONFIG)
clips=cut_hits(audio.y,audio.sr,detect_hits(audio.y,audio.sr,CONFIG),CONFIG)
assert clips,"Popraw segmentację w notebooku 01/02."
clip=clips[0]
noise=load_audio(BACKGROUND,CONFIG).y if BACKGROUND else None""",
        ),
        (
            "md",
            "## Kontrolowane zmiany\nBez pliku tła dodajemy syntetyczny biały szum. To test techniczny; prawdziwa fabryka ma inne widmo i zmienność. SNR = stosunek energii sygnału do tła w tym oknie, nie pomiar poziomu hałasu hali.",
        ),
        (
            "code",
            """from pinglab.robustness import sensitivity_table
from pinglab.plots import compare_hits
stats,variants=sensitivity_table(clip['y'],audio.sr,noise,clip['onset_offset_s'],FMAX)
display(stats[['variant','peak_hz','centroid_hz','decay10_s','flatness','clipping_fraction']].round(3))
display(compare_hits([variants['original'],variants['noise 10 dB']],audio.sr,
                     ['Oryginał','Szum 10 dB'],[clip['onset_offset_s']]*2,FMAX))
for name in ['original','gain -12 dB','noise 10 dB']:
    print(name)
    # Normalize only for convenient playback; metrics above use raw perturbations.
    display(listen(variants[name],audio.sr,normalize=True))""",
        ),
        (
            "md",
            "**Oczekiwane:** zmiana gainu prawie nie zmienia cech kształtu. Szum skraca wiarygodny fragment zaniku i zmienia barwę. Nie odszumiaj agresywnie „na wszelki wypadek”: możesz usunąć sygnał wady. Lepiej poprawić nagranie i porównać wyniki przed / po świadomie dobranym filtrowaniu.",
        ),
        (
            "md",
            "## Czy model przeżywa zmianę tła?\nTen test zostawia całe płytki poza treningiem. Porównuje trening na czystym audio i z dodanym tłem; powielone warianty nie zwiększają liczby płytek. Dane i tło pochodzą ze wspólnego `project.json`. Zatwierdź wcześniej czasy w 02. Biały szum nie odtwarza całej hali.",
        ),
        (
            "code",
            """from pinglab.dataset import analyze_manifest
from pinglab.robustness import noise_robustness
STRESS_MANIFEST = SESSION.manifest
STRESS_ANNOTATIONS = SESSION.annotations
stress_events, _ = analyze_manifest(STRESS_MANIFEST,SESSION.output/'stress',CONFIG,FMAX,
    STRESS_ANNOTATIONS if STRESS_ANNOTATIONS.exists() else None)
try:
    stress = noise_robustness(stress_events, family='resonance', noise=noise, config=CONFIG, fmax=FMAX, snrs=(20,10,5,0))
    display(stress.round(3))
    stress.to_csv(SESSION.output/'stress/robustness.csv',index=False)
except ValueError as error:
    print(error)""",
        ),
        (
            "md",
            "## Protokół na pierwszą serię\n1. Zapisz ID płytki, ocenę operatora, typ i miejsce wady; niepewne etykiety oznacz `unknown`.\n2. Ustal podparcie, mikrofon, gain i kolejność miejsc uderzeń. Nie zmieniaj ich między OK i NOK.\n3. Preferuj WAV PCM 48 kHz / 24 bit. Zostaw zapas amplitudy, wyłącz AGC i odszumianie komunikatora.\n4. Nagraj chwilę ciszy przed uderzeniem i pełne wybrzmiewanie po nim; odstęp ustal na pierwszym odsłuchu.\n5. Zbierz powtórzenia na różnych płytkach obu klas i osobne tło hali. Przeplataj OK i NOK, by czas/sesja nie zdradzały etykiety.\n6. Jeśli operator wskazuje jedyne słyszalne miejsce wady, zaznacz ten konkretny hit. Mediana z dziesięciu miejsc może go ukryć.\n7. Osobno sprawdź zmianę miejsca mikrofonu, siły uderzenia i hałasu. Nie mieszaj od razu wszystkich zmian.",
        ),
        (
            "md",
            "## Decyzja po pierwszych 30 płytkach\n**Kontynuuj**, jeżeli różnica powtarza się na niewidzianych detalach i ma sens akustyczny. **Popraw protokół**, jeśli separacja pokrywa się z sesją, głośnością lub podparciem. **Zbierz więcej danych**, jeśli błędy zależą od konkretnego typu wady. Uzgodnij z klientem osobno dopuszczalne przeoczenia NOK i odrzucenia OK.\n\nNie ma metody niezależnej od dowolnego nagrania. Ten zestaw pomaga szybko ustalić, co jest stabilne i czego jeszcze brakuje.",
        ),
    ],
)


write('00_eksploracja_do_przegladu.ipynb', [
('md', '''# Ping Lab · pierwszy przegląd nagrań
**Cel spotkania:** ustalić, czy w nagraniach widać i słychać powtarzalny sygnał wady oraz co poprawić w pomiarach.

**Start:** otwórz folder projektu w VS Code → wybierz kernel `.venv` / **Ping Lab (Python)** → **Run All**.
Wszystkie notebooki czytają `project.json`. Po otrzymaniu danych zmień tam manifest, typ danych i katalog wyników.
Każdy `plate_id` oznacza fizyczną płytę; kilka uderzeń nie zwiększa liczby niezależnych przykładów.

Ten plik nadaje się do przeglądu z zapisanymi wynikami. Do ponownego uruchomienia potrzebny jest dołączony projekt.'''),
('code', BOOT),
('md', '## 1. Co dostaliśmy?\nNiepewne oceny oznacz `unknown`. Typ płyty, temperatura i sesja pomagają odróżnić wadę od warunków nagrania.'),
('code', '''from pinglab.exploration import audio_gallery, spectrogram_gallery, feature_atlas, metadata_audit
from pinglab.dataset import analyze_manifest
OUT = SESSION.output / 'exploration'
records = SESSION.records
objects = records[records.label != 'background']
display(objects.groupby('label').agg(pliki=('path','size'), plyty=('plate_id','nunique')))
display(objects[['path','plate_id','label','session_id','plate_type','temperature']])
print('Źródło:', SESSION.source_kind, '· plików tła:', int(records.label.eq('background').sum()))'''),
('md', '## 2. Jakość i podział na uderzenia\nSprawdź przesterowanie, za ciche nagrania i liczbę kandydatów. „Auto” wymaga odsłuchu — mowa lub stuknięcie stołu także mogą przekroczyć próg. Korektę sekund zapiszesz w notebooku 02.'),
('code', '''events, quality = analyze_manifest(MANIFEST, OUT, CONFIG, FMAX,
    ANNOTATIONS if ANNOTATIONS.exists() else None)
display(quality[['path','native_sr','peak_dbfs','clipping_fraction','n_hits','reviewed','warnings']])
print('Uderzenia:', len(events), '· pliki bez uderzeń:', int(((quality.label!='background') & quality.n_hits.eq(0)).sum()))
if not events.empty:
    print('Skrócone okna:', int(events.truncated.sum()), '— sprawdź odstęp uderzeń i długość ogona.')'''),
('md', '## 3. Odsłuch każdego pliku\nRozwiń wybrane nagranie. Głośność oryginalna; różnica poziomu nie jest dowodem wady. Po odsłuchu zanotuj, jaką różnicę słyszy operator.'),
('code', 'audio_gallery(SESSION)'),
('md', '## 4. Galeria: sygnał i spektrogram\nPomarańczowe kreski to początki kandydatów. Jasne poziome pasma wskazują rezonanse, ich długość — wybrzmiewanie. Skala jest względna dla każdego pliku; kolory nie porównują bezwzględnej głośności.'),
('code', 'spectrogram_gallery(SESSION)'),
('md', '## 5. Czy cechy są powtarzalne?\nZobacz częstotliwość i zanik oraz rozrzut uderzeń na tej samej płycie. Duży rozrzut może oznaczać lokalną wadę, ale także zmienne miejsce lub siłę uderzenia. Zanik o 10 dB jest opisem okna, nie pomiarem stałej materiałowej.'),
('code', 'feature_atlas(events)'),
('md', '## 6. Czy porównujemy podobne warunki?\nJeżeli wszystkie NOK pochodzą z jednej sesji lub mają inny typ, model może rozpoznawać warunki zamiast pęknięcia.'),
('code', 'metadata = metadata_audit(SESSION)'),
('md', '''## 7. Pierwsze proste modele
Porównujemy **odległość od dobrych wzorców**, **regresję logistyczną na rezonansach i ich zmienności** oraz **MFCC + średnią z 3 najwyższych ocen uderzeń**.
Każda oceniana płyta jest poza treningiem. Przy małej próbie modelowanie zostanie pominięte.

Próg jest ustalany wewnątrz treningu, jeśli wystarcza danych; inaczej używamy jawnego progu stałego.
**Recall NOK** = część wykrytych wad. **False reject** = część dobrych płyt odrzuconych. Żaden wynik na demo nie potwierdza skuteczności u klienta.'''),
('code', '''from pinglab.benchmark import benchmark
leaderboard = predictions = folds = taps = None
if events.empty:
    print('Modelowanie pominięte: brak uderzeń.')
else:
    try:
        leaderboard, predictions, folds, taps = benchmark(events)
        display(leaderboard[['method','n_plates','recall_nok','false_reject','missed_nok','roc_auc','average_precision','recall_cp95']].round(3))
        display(predictions.groupby(['method','threshold_policy']).size().rename('płyty'))
        leaderboard.to_json(OUT/'leaderboard.json',orient='records',indent=2)
        predictions.to_csv(OUT/'predictions.csv',index=False)
        (OUT/'folds.json').write_text(json.dumps(folds,indent=2))
    except ValueError as error:
        print('Modelowanie pominięte:', error)'''),
('md', '''## 8. Wnioski do omówienia
Uzupełnij po odsłuchu prawdziwych nagrań:

| Pytanie | Nasza obserwacja |
|---|---|
| Czy nagrania nadają się do porównania? | … |
| Jaka różnica jest słyszalna i gdzie widać ją na wykresie? | … |
| Czy powtarza się na innych płytach tego samego typu? | … |
| Co zmienić w kolejnym nagraniu? | … |

**Kolejność:** 1 — jakość, odsłuch, podział, kilka cech i proste modele; 2 — szum i liczba uderzeń; 3 — zamrożony AST, jeśli potrzebny.
Szczegóły porównania i błędy każdej płyty: **06_porownanie_metod.ipynb**. Do korekty podziału: **02**.

Przy 10 wadliwych płytach nawet 10/10 trafień daje dolną granicę dwustronnego 95% przedziału dwumianowego około 69%. To nie gwarantuje wyników w nowej sesji.
**Run All zakończony** oznacza wykonanie analizy, a nie potwierdzenie jakości danych.'''),
('code', "print('GOTOWE · eksploracja · źródło:', SESSION.source_kind)")
])

write('06_porownanie_metod.ipynb',[
('md', '''# 06 · Porównanie metod i liczby uderzeń
**Cel:** porównać kilka ustalonych z góry metod na tych samych płytach. `project.json` wybiera dane.
A: średnia i odchylenie 3 cech rezonansu; B: 26 MFCC, model każdego uderzenia i średnia 3 największych ocen.
Odległość uczy się tylko na OK; klasyfikatory korzystają z OK i NOK. Każda płyta w treningu B ma tę samą łączną wagę.

Priorytetem jest liczba pominiętych NOK i odrzucanych OK. Nie uruchamiamy dziesiątek modeli, żeby wybrać przypadkowo najlepszy.'''),
('code', BOOT),
('code', '''from pinglab.dataset import analyze_manifest
from pinglab.benchmark import benchmark, tap_count_curve
import matplotlib.pyplot as plt
OUT = SESSION.output / 'comparison'
events, quality = analyze_manifest(MANIFEST,OUT,CONFIG,FMAX,
    ANNOTATIONS if ANNOTATIONS.exists() else None)
result = None
if events.empty:
    print('Brak uderzeń; popraw segmentację w 02.')
else:
    try:
        result = benchmark(events)
    except ValueError as error:
        print(error)'''),
('md', '''## Metryki poza treningiem
Podział: do 5 foldów, rozłączne fizyczne płyty. Skalowanie i model są liczone ponownie w treningu.
Jeśli w treningu jest przynajmniej 5 płyt każdej klasy, wewnętrzna walidacja dobiera próg bez pominięć NOK; zewnętrzny test może nadal mieć przeoczenia. Przy mniejszej próbie próg pozostaje stały.
`average_precision` podsumowuje krzywą precision–recall (nie pole liczone trapezami). Przedziały są opisowe; bootstrap ustalonych prognoz nie obejmuje niepewności treningu.'''),
('code', '''if result is not None:
    leaderboard,predictions,folds,taps = result
    display(leaderboard.round(3))
    leaderboard.to_json(OUT/'leaderboard.json',orient='records',indent=2)
    predictions.to_csv(OUT/'predictions.csv',index=False)
    (OUT/'folds.json').write_text(json.dumps(folds,indent=2))
    display(predictions.groupby(['method','threshold_policy']).size().rename('płyty'))'''),
('md', '## Każdy błąd osobno\nWiersze poniżej to płytki pominięte lub niepotrzebnie odrzucone. Wróć do ich odsłuchu i zdjęcia. Nie poprawiaj progu na tych samych testowych błędach i nie przedstawiaj potem wyniku jako niezależnego.'),
('code', '''if result is not None:
    marked=predictions.assign(predicted=np.where(predictions.score>=predictions.threshold,'nok','ok'))
    errors=marked[marked.label!=marked.predicted]
    metadata=SESSION.records[['plate_id','path','plate_type','temperature','photo','notes']]
    display(errors.merge(metadata,on='plate_id',how='left'))
    fig,axes=plt.subplots(1,len(leaderboard),figsize=(11,3.5),constrained_layout=True)
    for ax, row in zip(axes,leaderboard.itertuples()):
        cm=np.array(row.confusion_matrix)
        ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        ax.set(title=row.method,xticks=[0,1],xticklabels=['OK','NOK'],yticks=[0,1],yticklabels=['OK','NOK'],xlabel='Ocena modelu',ylabel='Etykieta')
    display(fig); plt.close(fig)'''),
('md', '''## Ile uderzeń potrzeba?
Dla modelu B losujemy bez zwracania 1/3/5/10 uderzeń, 100 razy. Używamy ocen z modelu, który nie widział tej płyty, i zachowujemy próg dobrany dla pełnego zestawu uderzeń.
Wszystkie k dotyczą tych samych płyt. Pomijamy k większe niż liczba dostępnych uderzeń na najuboższej płycie — demo ma tylko 3.
To wrażliwość na wybór uderzeń przy ustalonym modelu i progu, nie ponowny trening dla każdego k. Rozrzut losowań nie jest przedziałem skuteczności dla populacji.'''),
('code', '''if result is not None:
    curve=tap_count_curve(taps,repeats=100)
    curve.to_csv(OUT/'tap_count.csv',index=False)
    if not curve.empty:
        summary=curve.groupby('k')[['recall_nok','false_reject']].agg(['mean','min','max'])
        display(summary.round(3))
        fig,ax=plt.subplots(figsize=(8,3.5),constrained_layout=True)
        for metric,name in [('recall_nok','Wykryte NOK'),('false_reject','Odrzucone OK')]:
            q=curve.groupby('k')[metric]
            ax.plot(q.mean().index,q.mean(),marker='o',label=name)
            ax.fill_between(q.mean().index,q.quantile(.05),q.quantile(.95),alpha=.15)
        ax.set(xlabel='Liczba uderzeń',ylabel='Odsetek',ylim=(-.03,1.03),xticks=summary.index)
        ax.legend(); display(fig); plt.close(fig)
    print('Wyniki zapisane. Dla powtórnych sesji użyj także testu sesji w 03.')'''),
('md', '''## Co świadomie odkładamy?
AST jest opcjonalnym porównaniem w 04. Pełny trening sieci, rozbudowane strojenie SVM/lasów i SHAP nie są potrzebne do pierwszej oceny wykonalności.
Estymacja tłumienia każdego modu wymaga czystych nagrań i weryfikacji fizycznej — nie nazywamy prostego czasu zaniku pomiarem materiałowym.
**Wynik bez separacji też jest wynikiem PoC.** Najpierw popraw etykiety i protokół, potem dodawaj złożoność.'''),
('code', "print('GOTOWE · porównanie metod · źródło:', SESSION.source_kind)")
])
print('Created 7 notebooks')

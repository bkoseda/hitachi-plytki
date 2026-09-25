from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BOOT = """from pathlib import Path
import os, json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
assert (ROOT / "pyproject.toml").exists(), "Uruchom notebook z katalogu repo lub notebooks/."
os.chdir(ROOT)
from pinglab.audio import Config, load_audio, detect_hits, cut_hits, listen
from pinglab.demo import generate_demo
if not (ROOT / "data/demo/SYN_001_ok.wav").exists():
    generate_demo(ROOT / "data/demo")
CONFIG = Config()
FMAX = 20000  # wspólne pasmo wszystkich plików; dla źródeł 16 kHz ustaw np. 7500
print("Repo:", ROOT)
"""


def write(name, cells):
    nb = nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in cells]
    )
    nb.metadata = dict(
        kernelspec=dict(display_name="Ping Lab (Python)", language="python", name="pinglab"),
        language_info=dict(name="python", version="3.12"),
    )
    nbf.write(nb, ROOT / "notebooks" / name)


write(
    "01_pierwsza_para.ipynb",
    [
        (
            "md",
            "# 01 · Pierwsze dwa pliki\n**Cel:** odsłuchać, sprawdzić jakość i zobaczyć różnice. Około 10 minut.\n\nDomyślnie otwierasz **symulację**, nie pomiary klienta. Zmień tylko komórkę KONFIGURACJA. Pracujemy lokalnie. `Shift+Enter` uruchamia komórkę; **Run → Run All Cells** uruchamia całość.",
        ),
        ("code", BOOT),
        (
            "md",
            "## Wrzuć pliki\nOpcjonalny przycisk zapisze je do `data/raw/inbox/`. Możesz też skopiować je Finderem. Wklej ścieżki poniżej.",
        ),
        ("code", 'from pinglab.widgets import upload_box\nuploader = upload_box(ROOT / "data/raw/inbox")'),
        (
            "code",
            """# KONFIGURACJA — jedyne obowiązkowe zmiany po otrzymaniu plików
FILE_A = ROOT / "data/demo/SYN_001_ok.wav"
FILE_B = ROOT / "data/demo/SYN_021_nok.wav"
NAME_A, NAME_B = "A · demo OK", "B · demo NOK"
CHANNEL_A, CHANNEL_B = None, None  # None = kanał z najwyższym RMS; 0/1 wymusza kanał
MANUAL_A, MANUAL_B = None, None  # np. [0.4, 1.9, 3.4]; [] = brak uderzeń
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
MANIFEST = ROOT / "data/demo/manifest.csv"
ANNOTATIONS = ROOT / "data/demo/reviewed_annotations.json"
OUT = ROOT / "reports/runs/notebook_batch"
""",
        ),
        (
            "md",
            "## Manifest\nWiersz = plik. Obowiązkowe: `path`, `plate_id`, `label`, `session_id`. Etykiety: `ok`, `nok`, `unknown`, `background`. Ścieżka względna jest liczona **od pliku CSV**. Szablon: `data/manifest.template.csv`.\n\n`channel` = opcjonalny numer kanału (0/1, puste = auto). `position` = miejsce uderzenia; `environment` = warunki; `notes` = opis wady/podparcia; `photo` = ścieżka zdjęcia. Przy wielu miejscach w jednym pliku wpisz kolejność i zachowuj ten sam protokół.",
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
            "## Zatwierdź uderzenia\nWybierz plik → **Wczytaj / wykryj** → odsłuchaj → popraw sekundy → **Podgląd czasów** → **Zatwierdź i zapisz**. Większy próg wykrywa mniej kandydatów. Zatwierdzone czasy mają pierwszeństwo przed automatem; aby przeliczyć je automatycznie, usuń wpis z JSON. Puste pole i zapis oznaczają świadome odrzucenie wszystkich uderzeń.",
        ),
        (
            "code",
            "from pinglab.widgets import review_manifest\nreview = review_manifest(MANIFEST,ANNOTATIONS,CONFIG)",
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
source_kind = 'synthetic' if events.environment.eq('synthetic').any() else 'client'
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
MANIFEST = ROOT / "data/demo/manifest.csv"
ANNOTATIONS = ROOT / "data/demo/annotations.json"  # dla klienta: zatwierdzony plik z 02
OUT = ROOT / "reports/runs/notebook_models"
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
source_kind='synthetic' if events.environment.eq('synthetic').any() else 'client'
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
            "# 04 · Gotowy model audio, bez treningu sieci\n**Cel:** sprawdzić alternatywny opis dźwięku. AST zamienia uderzenie na 768 liczb. Używamy wag AudioSet w trybie zamrożonym. To model ogólnych zdarzeń audio, nie wyspecjalizowany detektor pęknięć.\n\nNa tym Macu model jest już pobrany. Obliczenia pozostają lokalne. CPU jest domyślny; 48 GB RAM wystarcza z dużym zapasem dla pojedynczych wywołań.",
        ),
        ("code", BOOT),
        (
            "code",
            """# KONFIGURACJA
MANIFEST = ROOT / "data/demo/manifest.csv"
ANNOTATIONS = ROOT / "data/demo/annotations.json"
OUT = ROOT / "reports/runs/notebook_ast"
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
FILE = ROOT / "data/demo/SYN_001_ok.wav"
BACKGROUND = None  # np. ROOT / "data/raw/factory_background.wav"
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
            "## Czy model przeżywa zmianę tła?\nTen test zostawia całe płytki poza treningiem. Porównuje trening na czystym audio i z dodanym tłem; powielone warianty nie zwiększają liczby płytek. To demonstracja na symulacji. Dla klienta ustaw własny manifest, zatwierdzone czasy i prawdziwe tło. Biały szum nie odtwarza całej hali.",
        ),
        (
            "code",
            """from pinglab.dataset import analyze_manifest
from pinglab.robustness import noise_robustness
STRESS_MANIFEST = ROOT / "data/demo/manifest.csv"
STRESS_ANNOTATIONS = ROOT / "data/demo/annotations.json"
stress_events, _ = analyze_manifest(STRESS_MANIFEST,ROOT/'reports/runs/stress',CONFIG,FMAX,
    STRESS_ANNOTATIONS if STRESS_ANNOTATIONS.exists() else None)
try:
    stress = noise_robustness(stress_events, family='resonance', noise=noise, config=CONFIG, fmax=FMAX)
    display(stress.round(3))
    stress.to_csv(ROOT/'reports/runs/stress/robustness.csv',index=False)
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
print("Created 5 notebooks")

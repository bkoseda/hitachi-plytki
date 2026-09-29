"""Build a compact feature/background report and a runnable diagnostic notebook."""

import base64

import mistune
import nbformat as nbf
import pandas as pd

from pinglab.clientstudy import OUT_REL, find_root

ROOT = find_root()
OUT = ROOT / OUT_REL / "explanations"

REPORT = """# Co wykorzystują modele i czy rozpoznają tło?

29.09.2026 · uzupełnienie analizy obu paczek

**Kierunek produkcyjny: MFCC + regresja logistyczna, porównywane z XGBoostem. AST pozostaje tylko punktem odniesienia.** Wyniki poniżej dotyczą pięciu zakładanych płytek, nie potwierdzonych mechanizmów wykrywania pęknięć.

## Połączenie MFCC + physics

Wspólny wektor ma 41 cech. W aktualnym porównaniu LR daje 27/27 poprawnych nagrań, XGB 26/27: tyle samo co samo MFCC. Physics nie poprawiło tutaj accuracy. W modelach łączonych przetasowanie całego MFCC obniża balanced accuracy średnio o około 41 punktów procentowych, a całego physics o około 5 pp (LR) lub 7 pp (XGB). Model korzysta więc z physics, ale główną rolę pełni MFCC. To diagnostyka sześciu współzależnych podziałów, nie dowód fizycznej przyczyny.

## Co dostaje model i jak się uczy

1. Z nagrania wykrywamy kandydatów na uderzenia. Każde zaakceptowane okno ma 170 ms: 20 ms przed początkiem i 150 ms po nim. Odejmujemy składową stałą i normalizujemy amplitudę.
2. Jedno uderzenie staje się jednym wierszem: **26 liczb MFCC** albo **15 cech physics**. MFCC to 13 współczynników opisujących kształt widma, każdy jako średnia i odchylenie standardowe po krótkich ramkach wewnątrz uderzenia. Nie używamy współczynnika c0, głównie opisującego energię.
3. Z każdego z 27 plików bierzemy 10 uderzeń rozłożonych w czasie: razem 270 wierszy. Etykieta pochodzi z płytki. Dwie klasy i poszczególne płytki mają zrównoważone wagi.
4. W każdej z sześciu prób uczymy na jednej NOK i dwóch OK; testujemy całą drugą NOK i jedną OK. Skaler uczy się tylko na treningu. Regresja: C=0.1. XGBoost: 60 drzew, głębokość 2, learning_rate=0.05, reg_lambda=10; parametry nie są optymalizowane na tym zbiorze.
5. Model ocenia **każde uderzenie osobno**. Regresja sumuje wystandaryzowane cechy pomnożone przez wagi i przepuszcza wynik przez sigmoidę. XGBoost sumuje wkład małych drzew z pytaniami o progi cech. Żaden nie analizuje kolejności tapów ani odstępów w serii.
6. Decyzja nagrania to mediana ocen jego 10 uderzeń, z progiem 0.5. W tabeli eksperymentu uśredniamy wyniki modeli z prób, w których plik był odłożony. Model wdrożeniowy trzeba później osobno dopasować i ocenić na nowej sesji; obecna ocena nie jest skalibrowanym prawdopodobieństwem pęknięcia.

## Które cechy są najistotniejsze

Na odłożonych płytkach podmieniamy jedną cechę między nagraniami, przenosząc cały profil 10 tapów. Pozostałe cechy pozostają bez zmian. Powtarzamy 30 razy na każdy podział. Jeśli model zaczyna popełniać większy błąd, ta cecha jest mu potrzebna. Ranking poniżej używa wzrostu **log-loss** — miary karzącej również błędną pewność. Podajemy też spadek balanced accuracy, czyli średniej skuteczności dla obu klas. Nie są to wyniki na danych treningowych.

**MFCC + regresja:** najważniejsze są `mfcc_02_mean`, `mfcc_13_mean`, `mfcc_10_std`, następnie `mfcc_09_mean` i `mfcc_10_mean`. **MFCC + XGBoost:** `mfcc_02_mean`, `mfcc_13_mean`, `mfcc_09_mean`, `mfcc_10_std`. Pierwsze cztery cechy XGB i pierwsze trzy LR zwiększały log-loss po zaburzeniu w każdym z sześciu podziałów. Podziały współdzielą płytki, więc nie są sześcioma niezależnymi dowodami.

`mfcc_02_mean` opisuje jeden z szerokich wzorców kształtu widma; `mfcc_13_mean` — drobniejsze zróżnicowanie tego kształtu. Numer MFCC nie odpowiada pojedynczej częstotliwości ani pasmu. `mfcc_10_std` mówi, jak bardzo dany opis zmienia się wewnątrz krótkiego dźwięku. Łącznie średnie MFCC są istotniejsze od ich zmienności w czasie. Nie można nazwać konkretnego MFCC „cechą pęknięcia”: mikrofon, podparcie, odszumianie i tło też zmieniają widmo.

**Physics + regresja:** płaskość widma (`flatness`), udział energii 3.18–7.97 kHz (`band_5`) i szerokość widma. **Physics + XGBoost:** udział 80–201 Hz (`band_1`), płaskość, udział 201–504 Hz (`band_2`) i 3.18–7.97 kHz (`band_5`). Płaskość jest większa, gdy energia jest bardziej równomiernie rozłożona, jak w szumie; mniejsza przy wyraźnych tonach. Pasma to udziały w energii analizowanego widma, nie surowa głośność. Niskie częstotliwości mogą również opisywać stanowisko lub odgłos samego uderzenia.

Cechy zaniku nie są obecnie głównym źródłem jakości XGB. To nie dowodzi, że tłumienie jest fizycznie nieistotne: krótkie okna, szybkie serie i możliwe wyciszanie ograniczają jego pomiar. Physics + LR ma słabszy wynik niż MFCC, więc jego ranking trzeba traktować szczególnie ostrożnie.

[IMPORTANCE]

Cechy bywają skorelowane. Zmiana jednej może zostać zastąpiona informacją z innej; podmieniona kombinacja cech może też być nietypowa. Ranking nie jest ani udziałem procentowym w decyzji, ani dowodem przyczynowym. Nie wybieraliśmy na jego podstawie nowej listy cech i nie ogłaszamy nowej skuteczności po dostrojeniu.

## Czy istnieje różnica niezwiązana z uderzeniem?

**Tak, mamy wyraźne poszlaki.** W poprzedniej rundzie sama reguła AAC=NOK / WAV=OK dawała 25/27 trafień. Model używający tylko długości, poziomu i wyciszeń, bez kodeka, miał 24/27. Poniżej dodatkowe eksperymenty; to osobne modele kontrolne trenowane od początku na tych samych podziałach po płytkach, a nie podawanie ciszy modelowi wytrenowanemu na pingach.

[CONTROLS]

- **100 ms przed uderzeniem** oznacza zakres od −120 do −20 ms. LR ma 24/27, XGB 25/27 trafień. Tło lub pozostałość poprzedniego pingu już zawiera informację o klasie. Przy odstępach około 0.28 s to nie jest czyste tło, więc nie przypisujemy całego wyniku zakłóceniom pomiaru.
- **Najcichsze 100 ms z dala od wykryć:** wybieramy do 5 rozłącznych okien na plik, co najmniej 400 ms po każdym kandydacie i 100 ms przed kolejnym. W 25 plikach znaleziono 118 okien; R03 i G01 nie mają odpowiednich fragmentów. XGB ma 22/25 trafień, balanced accuracy 81.25%, recall NOK 5/8. Zawsze-OK miałby 17/25, ale balanced accuracy tylko 50%. To mocniejszy sygnał różnic tła lub toru zapisu. Fragmenty wybrano automatycznie, mogą zawierać niewykryte dźwięki lub resztki wybrzmień; wymagają odsłuchu. Znaczna część najcichszych fragmentów OK jest cyfrowym zerem. Model nie dostaje osobnej kolumny głośności, ale może rozpoznać brak sygnału, kwantyzację czy kształt szumu.
- **Usunięcie 20 ms przed pingiem:** LR nadal ma 27/27. Samo ucięcie tej części nie usuwa problemu — tło, filtr i charakterystyka nagrywania są obecne także podczas uderzenia.
- **Tylko pierwsze 60 ms pingu:** LR ma 26/27. Krótki atak również przenosi użyteczną informację; nadal może ona dotyczyć siły, narzędzia lub filtracji. Nie jest to gotowy nowy wariant produkcyjny.

[BACKGROUND]

Nie ustalamy z tych prób „procentu decyzji wynikających z tła”. Pokazują, że zbiór ma skróty do etykiety i należy je wyeliminować w kolejnym pomiarze. Nowe nagrania oraz stare NOK należą do różnych paczek, a paczka, ustawienia, obiekt i klasa są częściowo splecione. Losowy podział tapów tylko ukryłby ten problem.

## Najkrótszy test, który pomoże rozstrzygnąć

1. Nagrać dostępne OK i NOK naprzemiennie, w jednej sesji, tym samym mikrofonem, aplikacją, gainem, WAV PCM i geometrią. Wyłączyć ulepszenia; potwierdzić faktyczne wejście mikrofonowe.
2. Dla każdej płytki: 10 s nieprzetworzonego tła, 10 tapów co 2–3 s w oznaczonych miejscach, 10 s tła. Zapisać plate_id i session_id. Zmienność tła i pozycje nie mogą pokrywać się z klasą.
3. Powtórzyć po ponownym ustawieniu stanowiska. Zamrozić model z pierwszej sesji i sprawdzić drugą bez dostrajania. To test odporności sesyjnej tych samych płytek; osobny test nowych fizycznych sztuk nadal jest konieczny.
4. Powtórzyć kontrolę samego tła. Jeżeli tło nadal przewiduje klasę, szukać różnicy toru albo sekwencji pomiaru. Można też domieszać to samo obce tło lub zamieniać tła między klasami i mierzyć zmianę decyzji, ale taki eksperyment nie odzyska fragmentów wyciętych przez bramkę szumów.

Nie naprawimy stratnego AAC przez zapis do WAV i nie udowodnimy braku różnic pomiaru samą normalizacją amplitudy. Obie klasy potrzebują wspólnych warunków. To teraz ważniejsze niż wybór regresja vs XGB.

Metoda interpretacji: [scikit-learn — permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), [ograniczenia interpretacji współczynników](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html). Wszystkie podane liczby policzono lokalnie z obu paczek klienta.
"""


def write_report():
    importance = pd.read_csv(OUT / "feature_importance.csv")
    results = pd.read_csv(OUT / "control_results.csv")
    top = pd.concat(
        [
            g[~g.feature.str.startswith("ALL")].nlargest(5, "logloss_increase")
            for _, g in importance.groupby("model")
        ]
    )[["model", "feature", "logloss_increase", "ba_drop", "positive_loss_folds"]]
    top.columns = ["Model", "Cecha", "Wzrost log-loss", "Spadek BA", "Dodatni wzrost w podziałach"]
    top["Spadek BA"] = top["Spadek BA"].map(lambda x: f"{100 * x:.1f} p.p.")
    names = {
        "pre_hit_100ms": "Przed pingiem −120…−20 ms",
        "quiet_100ms": "Ciche okna z dala od wykryć",
        "no_pre_150ms": "Ping 0…150 ms",
        "early_60ms": "Ping 0…60 ms",
        "late_90ms": "Ogon 60…150 ms",
    }
    comp = results[results.model.ne("always OK")][
        ["control", "model", "n", "accuracy", "balanced_accuracy", "recall_nok", "false_reject_ok"]
    ].copy()
    comp["control"] = comp.control.map(names)
    for col in ["accuracy", "balanced_accuracy", "recall_nok", "false_reject_ok"]:
        comp[col] = comp[col].map(lambda v: f"{v:.1%}")
    comp.columns = [
        "Fragment",
        "Model",
        "Pliki",
        "Accuracy",
        "Balanced accuracy",
        "Recall NOK",
        "Odrzucone OK",
    ]
    tables = {
        "[IMPORTANCE]": top.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}"),
        "[CONTROLS]": comp.to_html(index=False, border=0),
    }
    def markdown_table(frame):
        values = [[str(c) for c in frame.columns], ["---"] * len(frame.columns)]
        for row in frame.itertuples(index=False, name=None):
            values.append([f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
        return "\n".join("| " + " | ".join(row) + " |" for row in values)

    md = (
        REPORT.replace("[IMPORTANCE]", markdown_table(top))
        .replace("[CONTROLS]", markdown_table(comp))
        .replace("[BACKGROUND]", "")
    )
    (ROOT / "docs/CLIENT_FEATURES_AND_BACKGROUND.md").write_text(md)
    body = mistune.html(REPORT)
    for marker, table in tables.items():
        body = body.replace("<p>" + marker + "</p>", table)
    for marker, filename in [("[IMPORTANCE]", "importance.png"), ("[BACKGROUND]", "background_levels.png")]:
        encoded = base64.b64encode((OUT / filename).read_bytes()).decode()
        img = f'<img src="data:image/png;base64,{encoded}" alt="{filename}">'
        if marker == "[IMPORTANCE]":
            body = body.replace("</table>", "</table>" + img, 1)
        else:
            body = body.replace("<p>" + marker + "</p>", img)
    css = "body{background:#f4f3ef;color:#20303a;font:17px/1.65 system-ui;margin:0}main{max-width:1100px;margin:35px auto;padding:40px;background:white}h1{font-size:36px;line-height:1.2}h2{margin-top:45px;border-top:1px solid #ddd;padding-top:20px}strong,a{color:#176e76}img{max-width:100%;height:auto}table{display:block;overflow:auto;border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}th{background:#eaf1f1}code{font-size:.92em}@media(max-width:700px){main{margin:0;padding:20px}}"
    (OUT / "START.html").write_text(
        f'<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cechy i tło — Ping Lab</title><style>{css}</style><main><a href="../START.html">← Raport obu paczek</a>{body}</main></html>'
    )
    return OUT / "START.html"


def notebook():
    cells = [
        (
            "m",
            "# Co wykorzystują regresja i XGBoost?\nUruchom **Run All** w `.venv`. Ten notebook nie uruchamia AST. Wymaga wyników poprzedniej analizy obu paczek. Porównuje ważność cech na odłożonych płytkach i osobne modele kontrolne uczone na tle. Nie dobiera automatycznie nowego modelu produkcyjnego.",
        ),
        (
            "c",
            """from pathlib import Path
import sys, json
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'config/client_archive_manifest.json').exists()), None)
if ROOT is None: raise FileNotFoundError('Otwórz folder ceramic-ping-lab.')
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown, Audio
from pinglab.clientstudy import OUT_REL, decode, load
from pinglab.clientexplanations import importances, controls
OUT = ROOT/OUT_REL/'explanations'
%matplotlib inline
RECOMPUTE = True
print(sys.executable)""",
        ),
        (
            "m",
            "## 1. Jedno uderzenie → cechy → ocena → mediana serii\n170 ms = 20 ms przed +150 ms po początku. Jeden wiersz ma 26 MFCC, 15 physics albo ich połączenie: 41 cech. Każdy model uczy się etykiet uderzeń na trzech płytkach; dwie całe inne płytki są testowe. Powtarzamy sześć podziałów. Dziesięć uderzeń na plik daje 270 wierszy. Seria nie trafia do modelu jako sekwencja: agregujemy dopiero jego oceny. AST jest wyłącznie punktem odniesienia w notebooku 03.",
        ),
        (
            "c",
            """splits = json.loads((ROOT/OUT_REL/'split_audit.json').read_text())
display(pd.DataFrame(splits)[['fold','train_groups','test_groups','train_taps','test_taps']])""",
        ),
        (
            "m",
            "## 2. Co się psuje po zaburzeniu jednej cechy?\nPrzenosimy cechę między całymi profilami nagrań testowych, po 10 tapów, 30 losowań. Mierzymy wzrost log-loss i spadek balanced accuracy. Cechy skorelowane mogą się zastępować, a nietypowe kombinacje mogą zawyżać wpływ. Ranking opisuje model w tym zbiorze, nie fizyczną przyczynę pęknięcia. Wyniki podziałów nie są niezależne.",
        ),
        (
            "c",
            """if RECOMPUTE or not (OUT/'feature_importance.csv').exists():
    importance = importances(ROOT, repeats=30)
else:
    importance = pd.read_csv(OUT/'feature_importance.csv')
display(Image(filename=str(OUT/'importance.png')))""",
        ),
        (
            "m",
            "## 3. Ranking i znaczenie nazw\nDla modeli łączonych pokazujemy również przetasowanie całej rodziny: ALL_mfcc i ALL_physics. To sprawdza wkład rodziny przy zachowaniu drugiej; pojedyncze skorelowane cechy mogą się zastępować.\n\nMFCC_mean to średnia po ramkach wewnątrz uderzenia, MFCC_std — zmienność. Numer MFCC nie jest numerem pasma Hz. `flatness`: podobieństwo widma do równomiernego szumu; `band_1`: udział energii 80–201 Hz, `band_2`: 201–504 Hz, `band_5`: 3.18–7.97 kHz. W tabeli LR dodatnia waga przesuwa ocenę ku NOK, ujemna ku OK (przy pozostałych cechach stałych). Gain XGB nie ma takiego znaku. Nie porównuj wartości gain bezpośrednio z wagą regresji.",
        ),
        (
            "c",
            """for name, g in importance.groupby('model'):
    display(Markdown('**'+name+'**'))
    display(g[~g.feature.str.startswith('ALL')].nlargest(7,'logloss_increase')[['feature','logloss_increase','ba_drop','positive_loss_folds','mean_abs_weight_or_gain','mean_signed_weight_or_gain']].round(3))
display(importance[importance.feature.str.startswith('ALL')][['model','feature','logloss_increase','ba_drop']].round(3))""",
        ),
        (
            "m",
            "## 4. Czy klasę można odgadnąć bez pingu?\nTo osobne modele uczone od początku. Pre-hit: −120…−20 ms, może zawierać poprzedni ogon. Quiet: do 5 najcichszych okien 100 ms, co najmniej 400 ms po wykryciu i 100 ms przed następnym. Nie ma takich fragmentów w R03/G01; pozostaje 25 plików. Okna mogą zawierać niewykryte dźwięki. Cyfrowa cisza jest reprezentowana jako brak sygnału, nie pomijana. Inne kontrole wycinają tylko 0–150, 0–60 lub 60–150 ms po pingu. Wszystkie zachowują podział po płytkach.",
        ),
        (
            "c",
            """if RECOMPUTE or not (OUT/'control_results.csv').exists():
    results = controls(ROOT)
else:
    results = pd.read_csv(OUT/'control_results.csv')
display(results[['control','model','n','n_nok','accuracy','balanced_accuracy','recall_nok','false_reject_ok']].round(3))
display(Image(filename=str(OUT/'background_levels.png')))""",
        ),
        (
            "m",
            "## 5. Audyt fragmentów tła\nKażdy wybrany fragment ma zapisane czasy. Poniżej po jednym oknie dwóch różnych plików; źródłowy poziom jest zachowany, więc może być niesłyszalny. Pełny odsłuch z przesuwaniem okna jest w notebooku 02. Sprawdź również, czy przed i po wybranym oknie nie ma pominiętego uderzenia.",
        ),
        (
            "c",
            """audit = pd.read_csv(OUT/'quiet_window_audit.csv')
display(audit.groupby(['group','recording_id']).agg(okna=('event_id','size'),RMS_dBFS=('rms_dbfs','median'),zera=('zero_fraction','median')).round(3))
records, _, _ = load(ROOT)
for rid in ['R02','G06']:
    selected = audit[audit.recording_id.eq(rid)].iloc[0]
    r = records.set_index('recording_id').loc[rid]
    x, sr = decode(ROOT/r.path)
    z = x[round(selected.start_s*sr):round(selected.end_s*sr), int(r.selected_channel)]
    display(Markdown(f"**{rid}: {selected.start_s:.2f}–{selected.end_s:.2f} s**"))
    display(Audio(z,rate=sr,normalize=False))""",
        ),
        (
            "m",
            "## 6. Co z tego robimy?\n**MFCC + regresja jako podstawowy kandydat, XGBoost jako porównanie.** Ranking nie jest dowodem, że model rozpoznaje pęknięcie. Nie wybieramy cech po obejrzeniu tych wyników i nie nazywamy kolejnego wyniku niezależnym testem.\n\nNastępny pomiar: te same ustawienia dla OK i NOK, naprzemiennie, 10 s tła przed/po, 10 tapów co 2–3 s. Powtórka po ustawieniu stanowiska od nowa. Model z pierwszej sesji testujemy na drugiej bez dostrajania, a potem na nowych fizycznych sztukach. Kontrola samego tła ma przestać zdradzać klasę.",
        ),
        (
            "c",
            """import importlib.util
spec = importlib.util.spec_from_file_location('feature_review',ROOT/'scripts/build_feature_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
print('Raport:',review.write_report())
from datetime import datetime, timezone
(OUT/'last_run.json').write_text(json.dumps({'finished_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'recompute':RECOMPUTE},indent=2))
print('GOTOWE — cechy i tło: wszystkie komórki zakończone.')""",
        ),
    ]
    nb = nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(s) if k == "m" else nbf.v4.new_code_cell(s) for k, s in cells]
    )
    nb.metadata.kernelspec = {"name": "pinglab", "language": "python", "display_name": "Ping Lab (Python)"}
    nb.metadata.language_info = {"name": "python", "version": "3.12.14"}
    nbf.write(nb, ROOT / "notebooks/client/04_cechy_i_tlo.ipynb")


if __name__ == "__main__":
    print(write_report())
    notebook()

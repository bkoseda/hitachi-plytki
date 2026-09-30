"""Reproducible report and notebook for September 30; no external services."""

import os
from html import escape
from pathlib import Path
from urllib.parse import quote

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/runs/client-update-20260930"
QUESTIONS = """1. Czy nowe `BAD/BAD2` to ta sama fizyczna płytka co wcześniejsze `Bad_2`, czy nowa pęknięta sztuka? Prosimy o krótkie przypisanie: folder → ID płytki → pęknięcie / wyszczerbienie / OK.
2. Czy pliki w `LAB TEST` powstały przed czy po wyłączeniu ulepszeń? Czy `LAB` oznacza inne pomieszczenie, aplikację lub mikrofon? W każdym z tych 13 plików nadal są długie odcinki dokładnego zera.
3. Czy `OK` z LAB i `OK` z Mic Enhancements Off to ta sama płytka? Która z poprzednich OK? Czy końcówki 2 i 3 oznaczają powtórzenie nagrania, a nie nową sztukę?
4. Co oznacza `angle` (kąt mikrofonu czy uderzenia) oraz `single point / one point` (miejsce uderzania czy podparcie)? W tych ZIP-ach nie znaleźliśmy nazw `no freq` ani `no frequency`."""
INTRO = """Nowe dane są wyraźnie inne. Folder **Mic Enhancements Off** wygląda lepiej pod kątem zachowania tła: 0/3 plików ma ciąg dokładnych zer dłuższy niż 100 ms. W **LAB TEST** takie odcinki występują w 13/13 plików, trwają do około 2,4 s; mediana udziału zer w pliku to około 56%. Sam sygnał nie rozstrzyga, czy to bramka, odszumianie, edycja czy przerwy toru.

Uderzenia są teraz rzadsze: typowy odstęp wynosi około 2,1–2,2 s wobec około 0,28 s wcześniej. To daje miejsce na ogon dźwięku, ale cyfrowe wyciszanie nadal ogranicza badanie zaniku. Nie mamy jeszcze dowodu, że dłuższy ogon poprawi klasyfikację.

**Najważniejszy wynik:** model MFCC + regresja uczony tylko na wcześniejszych danych trafia 12/13 plików LAB, lecz odrzuca oba nowe OK z Enhancements Off. Potrzebujemy NOK nagranego w tych samych warunkach z wyłączonymi ulepszeniami. Nie należy cofać tej zmiany po to, żeby poprawić wynik starego modelu."""
METHOD = """Analizujemy wszystkie 61 plików w kontroli jakości. Model pęknięć obejmuje 42 nagrania z roboczymi etykietami OK/NOK; tło, OLD o nieznanej tożsamości i Bitten są poza głównym treningiem. To nie oznacza, że wyszczerbienie jest niewykrywalne — jest innym rodzajem defektu.

Detektor szuka nagłych wzrostów krótkookresowego RMS: ramka 10 ms, krok 5 ms, próg zależny od tła, minimalny odstęp 120 ms. Odrzuca zbyt słabe i niepełne okna. W nowych danych przyjął 258 uderzeń; nie traktujemy ich jako ręcznie potwierdzonych trafień. Model dostaje do 10 równomiernie wybranych uderzeń z nagrania: każde to 20 ms przed + 150 ms po początku, z normalizacją amplitudy. Decyzja o nagraniu to mediana wyników uderzeń, próg 0,5.

Porównujemy physics (15 cech), MFCC (26), ich połączenie (41), zamrożony AST (768). LR: C=0,1. XGBoost: 60 drzew, głębokość 2, bez strojenia na nowych danych. Skalowanie i model uczą się tylko na części treningowej. Kontrola QC używa jedynie poziomu, udziału ciszy i długości pliku, bez widma i kodeka.

W teście łączonym odkładamy całe grupy uznane w danym scenariuszu za płytki: jedną NOK i jedną OK. Dla powtarzających się odłożeń uśredniamy wynik nagrania. Są to wyniki eksploracyjne, zależne od niepotwierdzonych ID; wiele nagrań tej samej sztuki nie jest wieloma niezależnymi płytkami. Scenariusze nie wyczerpują wszystkich możliwych przypisań. Nowe OK roboczo łączymy z OK_1, ale klient może wskazać inne ID."""


def summary_tables():
    quality = pd.read_csv(OUT / "batch_quality.csv")
    old = pd.read_csv(OUT / "old_to_new_metrics.csv")
    combined = pd.read_csv(OUT / "combined_metrics.csv")
    combined = combined[combined.level.eq("recording")]
    return quality, old, combined


def build_report():
    q, m, c = summary_tables()
    records = pd.read_csv(OUT / "manifest.csv")
    metrics = ["model", "n", "accuracy", "balanced_accuracy", "tp", "fn", "tn", "fp", "auc"]

    def table(df):
        return (
            '<div class="scroll">'
            + df.to_html(index=False, float_format=lambda x: f"{x:.3f}", escape=True)
            + "</div>"
        )

    def paras(txt):
        # Static author text only; preserve Markdown separately in versioned report.
        return "".join(
            "<p>" + escape(p).replace("**", "").replace("`", "") + "</p>" for p in txt.split("\n\n")
        )

    cards = []
    for r in records.itertuples():
        audio = quote(os.path.relpath(ROOT / r.path, OUT), safe="/")
        cards.append(
            f'<details><summary>{r.recording_id} · {escape(r.filename)}</summary><p>{r.duration_s:.1f} s · {r.n_accepted} przyjętych uderzeń · najdłuższe zero {r.longest_zero_s:.3f} s</p><audio controls preload="none" src="{audio}"></audio><img loading="lazy" src="figures/{r.recording_id}.png" alt="Przebieg i spektrogram {r.recording_id}"></details>'
        )
    pivot = c.pivot(index="model", columns="scenario", values="balanced_accuracy").reset_index()
    from pinglab.clientupdate import SCENARIOS

    scenario_html = "".join(f"<li><b>{k}</b>: {escape(v)}</li>" for k, v in SCENARIOS.items())
    body = f"""<h1>Nowe paczki · 30 września</h1><p>61 plików razem · 16 nowych · 42 nagrania w głównym porównaniu modeli</p>
<nav><a href="#quality">Jakość</a><a href="#models">Modele</a><a href="#questions">Do klienta</a><a href="#audio">Odsłuch</a></nav>
{paras(INTRO)}<section id="quality"><h2>Co zmieniło się w pomiarach</h2>{table(q)}<p>files_zero100ms: liczba plików z ciągiem dokładnego zera &gt;100 ms. median_floor_dbfs: dolny poziom RMS; −240 oznacza cyfrowe zero w obliczeniach. Brak twardego przesterowania nie wyklucza limitera. Wszystkie nowe pliki: 48 kHz, PCM16, dwa identyczne kanały.</p><img src="figures/quality.png"><img src="figures/envelopes.png"><img src="figures/background.png"><p>Nie jest to kontrolowane porównanie tego samego uderzenia przed/po przełączniku. Mikrofon, ustawienie i tożsamość płytki nie są potwierdzone. Brak długich zer nie dowodzi braku wszelkiego przetwarzania.</p></section>
<section id="models"><h2>Model z wcześniejszych danych → nowe nagrania</h2>{table(m[metrics])}<p>tp = wykryte NOK; fn = przeoczone NOK; tn = poprawnie przyjęte OK; fp = błędnie odrzucone OK. Balanced accuracy to średnia wykrywalności NOK i poprawnego przyjmowania OK. AUC mierzy kolejność wyników bez wyboru progu. To nowe nagrania, nie potwierdzony test nowych płytek.</p>
<h3>Oddzielnie dla obu dostaw</h3>{table(pd.read_csv(OUT / "old_to_new_by_batch.csv")[["model", "batch", "n", "tp", "fn", "tn", "fp", "auc"]])}<p>Enhancements Off zawiera tylko OK i tło: nie można obliczyć wykrywalności NOK. Wszystkie modele akustyczne mylą oba OK. W LAB wszystkie pliki mają ten sam format PCM, więc sam kodek nie wystarcza do ich rozdzielenia; nie wyklucza to różnic tła i sposobu pomiaru.</p>
<h2>Łącznie ze starszymi danymi: wpływ ID</h2><ul>{scenario_html}</ul>{table(pivot)}<p>MFCC + LR daje balanced accuracy około 0,84–0,96 zależnie od scenariusza. To poszlaka użytecznego sygnału, ale zakres scenariuszy nie jest przedziałem ufności. AST nie jest tutaj lepszym wyborem. QC bez analizy widma osiąga około 0,50–0,69, więc nadal sprawdzamy ryzyko różnic warunków. Wyniku łączonego nie należy porównywać wprost z testem starego modelu na nowej dostawie.</p>{paras(METHOD)}</section>
<section id="questions"><h2>Krótka lista do potwierdzenia z klientem</h2><ol>{"".join("<li>" + escape(x[3:]).replace("`", "") + "</li>" for x in QUESTIONS.splitlines())}</ol><h3>Najbardziej wartościowe dogranie</h3><p>Po jednym krótkim nagraniu OK i każdej rzeczywiście pękniętej płytki przy tym samym mikrofonie, aplikacji, podparciu i wyłączonych ulepszeniach. Kilka uderzeń co około 2 s, kilka sekund tła przed i po, WAV bez dodatkowej obróbki. Nazwy wystarczą proste: ID płytki + OK/pęknięcie/uszczerbienie + ustawienia off. Nie potrzeba rozbudowanego formularza.</p></section>
<section id="audio"><h2>Wszystkie nowe nagrania</h2><p>Zielone linie: zaakceptowane początki; złote: odrzucone kandydaty. Linie nie są pomiarem czasu zaniku.</p>{"".join(cards)}</section>
<footer>Notebook: notebooks/client/05_nowe_paczki_20260930.ipynb · Oryginały: assets/3-batch-enhancements-off i assets/4-batch-lab-test. Raport czyta audio z repo; nie jest samodzielną paczką. Bez wysyłania danych do usług zewnętrznych.</footer>"""
    css = """body{font:17px/1.6 system-ui;color:#233b42;background:#f6f4ef;margin:0}main{max-width:1120px;padding:38px 26px;margin:auto}h1{font-size:38px;line-height:1.15}h2{margin-top:35px}a{color:#087c83}nav{display:flex;gap:24px}section{border-top:1px solid #ccd7d5;margin-top:35px}img{width:100%;background:white}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#e0e9e5}.scroll{overflow-x:auto}details{background:white;padding:15px;margin:12px 0}summary{cursor:pointer}audio{width:100%}footer{margin-top:40px;font-size:14px;color:#53696e}li{margin:12px 0}"""
    (OUT / "START.html").write_text(
        f'<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Ping Lab · nowe paczki 30.09</title><style>{css}</style><main>{body}</main></html>'
    )
    report = (
        "# Nowe paczki klienta · 30.09.2026\n\n"
        + INTRO
        + "\n\n## Jak testujemy\n\n"
        + METHOD
        + "\n\n## Do potwierdzenia\n\n"
        + QUESTIONS
        + "\n\nPełne tabele i odsłuch: `reports/runs/client-update-20260930/START.html`. Notebook: `notebooks/client/05_nowe_paczki_20260930.ipynb`. Wyniki CSV w tym samym katalogu co HTML. Oryginały w `assets/3-batch-enhancements-off` i `assets/4-batch-lab-test`; wszystkie cztery dostawy są pod wspólnym `assets`.\n"
    )
    (ROOT / "docs/CLIENT_UPDATE_20260930.md").write_text(report)
    # Small versioned evidence table; regenerates from full, locally ignored outputs.
    (ROOT / "docs/CLIENT_UPDATE_20260930_RESULTS.csv").write_text(
        pd.concat([m.assign(test="old_to_new"), c.assign(test="combined")], ignore_index=True).to_csv(
            index=False
        )
    )


def build_notebook():
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        md("""# Nowe nagrania klienta · 30.09.2026
Otwórz repo w VS Code, kernel **.venv / Ping Lab**, kliknij **Run All**. Wszystko działa lokalnie. AST korzysta z wag `models/ast` i cache; pierwsze przeliczenie bez cache trwa dłużej. Na tym komputerze wagi są już pobrane.

Ten notebook jest aktualnym wejściem do analizy **wszystkich czterech dostaw**. Starsze 02–04 zachowują wyniki z 29.09. ID płytek w tej rundzie są założeniami do potwierdzenia."""),
        code("""from pathlib import Path
import sys
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/pinglab').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Otwórz notebook wewnątrz repo ceramic-ping-lab.')
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import pandas as pd
from IPython.display import display, Image, Markdown
from pinglab import clientupdate as update, clientstudy
from build_client_update import build_report, INTRO, METHOD, QUESTIONS
OUT = ROOT/update.OUT_REL
%matplotlib inline
records, events, features = update.prepare(ROOT)
display(Markdown(INTRO))"""),
        md(
            "## 1. Co przyjechało i jak interpretujemy nazwy\nKońcówki 2/3 traktujemy jako numery nagrań, nie dowód nowej płytki. „Mic Enhancements Off” jest deklaracją z nazwy. „LAB TEST” nie potwierdza wyłączenia ulepszeń. „single point” nie rozstrzyga rodzaju podparcia."
        ),
        code("""new = records[records.recording_id.str.startswith(('E', 'L'))]
display(new[['recording_id','filename','group','label','processing','duration_s','n_accepted']])
display(Markdown(QUESTIONS))"""),
        md(
            "## 2. Jakość: nowe kontra poprzednie\nDokładne zera liczymy w oryginalnym kanale, przed usuwaniem DC i normalizacją. Pojedyncze zerowe próbki PCM są normalne; sprawdzamy długie ciągi. −240 dBFS w tabeli oznacza cyfrowe zero; wykres ucina skalę dla czytelności."
        ),
        code("""update.figures(ROOT)
display(pd.read_csv(OUT/'batch_quality.csv').round(3))
display(Image(filename=str(OUT/'figures/quality.png')))
display(Image(filename=str(OUT/'figures/envelopes.png')))
display(Image(filename=str(OUT/'figures/background.png')))"""),
        md(
            "## 3. Posłuchaj i sprawdź segmentację\nUstaw RECORDING_ID, START_S i DURATION_S w poniższej komórce i uruchom ją. Odtwarzacz działa bez dodatkowych widgetów JavaScript. Zielone pasy to stałe okna cech, złote linie — odrzucone wykrycia. Odstęp uderzeń jest dłuższy, ale okno modelu pozostaje 170 ms, żeby zachować porównanie z poprzednią rundą. Odsłuch ma oryginalny poziom, bez automatycznego pogłaśniania."
        ),
        code("""RECORDING_ID = "L02"  # E01–E03 lub L01–L13; wybierz z tabeli powyżej
START_S = 0
DURATION_S = 7
clientstudy.inspect_recording(RECORDING_ID, START_S, DURATION_S, ROOT, out_rel=update.OUT_REL)"""),
        md("## 4. Uczenie i reprezentacja\n" + METHOD),
        code("""representations = update.representations(ROOT)
old_to_new, combined = update.model(ROOT)
display(representations.groupby(['batch','scope']).agg(uderzenia=('event_id','size'),nagrania=('recording_id','nunique')))
print('Żaden model nie jest strojony na nowej dostawie; warianty i próg są ustalone.')"""),
        md(
            "## 5. Czy stary model radzi sobie z nową dostawą?\nTrening: tylko poprzednie 27 nagrań głównych. Test: 15 nowych; nowe tło nie ma etykiety OK/NOK. To rozdział po dostawie, nie potwierdzony test nowych fizycznych płytek. tp/fn = wykryte/przeoczone NOK, tn/fp = przyjęte/odrzucone OK."
        ),
        code("""cols = ['model','n','accuracy','balanced_accuracy','recall_nok','tp','fn','tn','fp','auc']
display(old_to_new[cols].round(3))
display(pd.read_csv(OUT/'old_to_new_by_batch.csv')[['model','batch','n','tp','fn','tn','fp','auc']].round(3))
display(Markdown('**Oba OK z Enhancements Off są odrzucane przez modele akustyczne.** Nie mamy NOK z tej samej paczki. NaN w metrykach jednej klasy oznacza brak możliwości oceny, nie wynik zero.'))"""),
        md(
            "## 6. Modele na połączonych danych — cztery założenia o ID\nNie wybieramy najkorzystniejszego scenariusza. Klient powinien wskazać właściwe przypisanie; jeśli jest inne, przeliczamy podział. Zmiana scenariusza potrafi też zmienić zestaw ocenianych plików, więc zakres wyników nie jest przedziałem ufności."
        ),
        code("""display(Markdown('\\n'.join(f'- **{key}**: {value}' for key,value in update.SCENARIOS.items())))
r = combined[combined.level.eq('recording')]
display(r.pivot(index='model',columns='scenario',values='balanced_accuracy').round(3))
display(r[['scenario',*cols]].round(3))
# Pełne definicje train/test: identity_scenarios.json; wyniki każdego odłożenia: folds_*.csv."""),
        md(
            "## 7. Co dalej\nNajwięcej wniesie para OK/NOK w jednym ustawieniu z ulepszeniami off i potwierdzonymi ID. Kilka uderzeń co około 2 s, WAV bez obróbki, kilka sekund tła. Nie potrzeba teraz rozbudowanego formularza. Bitten i OLD zachowujemy do osobnego przeglądu; ich wynik nie mierzy skuteczności wykrywania pęknięć."
        ),
        code("""display(pd.read_csv(OUT/'excluded_diagnostic_predictions.csv').round(3))
build_report()
display(Markdown(QUESTIONS))
print('Raport:', OUT/'START.html')
print('Nowe oryginały:', ROOT/'assets/3-batch-enhancements-off', ROOT/'assets/4-batch-lab-test')"""),
    ]
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Ping Lab (Python)", "language": "python", "name": "pinglab"},
            "language_info": {"name": "python", "version": "3.12.14"},
        },
    )
    nbf.write(nb, ROOT / "notebooks/client/05_nowe_paczki_20260930.ipynb")


if __name__ == "__main__":
    build_report()
    build_notebook()

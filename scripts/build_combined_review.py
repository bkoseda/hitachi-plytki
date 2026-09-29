"""Build the Polish report and the two presentation notebooks from computed results."""

import base64
import html
import json
import os
from urllib.parse import quote

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import soundfile as sf

from pinglab.clientstudy import OUT_REL, find_root

ROOT = find_root()
OUT = ROOT / OUT_REL


def table(df):
    return df.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}", escape=True)


def picture(name):
    data = base64.b64encode((OUT / "figures" / name).read_bytes()).decode()
    return f'<img class="chart" src="data:image/png;base64,{data}" alt="{html.escape(name)}">'


def performance_figures():
    scores = pd.read_csv(OUT / "model_comparison.csv")
    s = scores[scores.level == "recording"].set_index("model")
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    names = [
        "physics + LR",
        "physics + XGB",
        "MFCC + LR",
        "MFCC + XGB",
        "MFCC + physics + LR",
        "MFCC + physics + XGB",
        "AST + LR",
        "AST + XGB",
        "QC only + LR",
    ]
    x = np.arange(len(names))
    ax.bar(x - 0.18, s.loc[names, "recall_nok"], 0.36, label="Recall NOK", color="#ba593d")
    ax.bar(x + 0.18, 1 - s.loc[names, "false_reject_ok"], 0.36, label="Poprawne OK", color="#167f86")
    ax.set(
        xticks=x,
        xticklabels=names,
        ylim=(0, 1.12),
        ylabel="Odsetek nagrań",
        title="9 nagrań NOK + 18 OK · tylko 5 zakładanych płytek",
    )
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.savefig(OUT / "figures/models.png", dpi=150)
    plt.close(fig)
    taps = pd.read_csv(OUT / "tap_count_resampling.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for model, g in taps.groupby("model"):
        for ax, col in zip(axs, ["recall_nok", "false_reject_ok"]):
            a = g.groupby("k")[col]
            ax.plot(a.mean().index, a.mean(), marker="o", label=model)
            ax.fill_between(a.mean().index, a.quantile(0.1), a.quantile(0.9), alpha=0.12)
    for ax, title in zip(axs, ["Wykryte nagrania NOK", "Odrzucone nagrania OK"]):
        ax.set(title=title, xlabel="Liczba uderzeń w decyzji", ylim=(-0.03, 1.03), xticks=[1, 3, 5, 10])
        ax.legend(fontsize=8)
    fig.savefig(OUT / "figures/tap_count.png", dpi=150)
    plt.close(fig)


def write_report():
    facts = json.loads((OUT / "quality_facts.json").read_text())
    m = pd.read_csv(OUT / "manifest.csv")
    scores = pd.read_csv(OUT / "model_comparison.csv")
    rec = scores[scores.level == "recording"]
    comp = rec[
        ["model", "accuracy", "balanced_accuracy", "recall_nok", "false_reject_ok", "auc", "tp", "fn", "fp"]
    ].copy()
    comp.columns = [
        "Model",
        "Accuracy",
        "Balanced accuracy",
        "Recall NOK",
        "Fałszywe odrzucenia OK",
        "AUC",
        "Trafione NOK",
        "Pominięte NOK",
        "Odrzucone OK",
    ]
    for c in ["Accuracy", "Balanced accuracy", "Recall NOK", "Fałszywe odrzucenia OK"]:
        comp[c] = comp[c].map(lambda v: f"{v:.1%}")
    codec = json.loads((OUT / "codec_control.json").read_text())
    noise = pd.read_csv(OUT / "noise_metrics.csv")
    noise = noise[["model", "variant", "recall_nok", "false_reject_ok", "accuracy"]]
    folds = pd.read_csv(OUT / "fold_metrics.csv")
    mainfolds = folds[folds.model == "MFCC + LR"]
    permutations = pd.read_csv(OUT / "group_label_permutations.csv")
    p = float((permutations.auc >= 1 - 1e-12).mean())
    md = f"""# Ocena nagrań płytek OK i NOK

29.09.2026 · dwie paczki klienta · analiza lokalna

## Najważniejsza decyzja

**W tej paczce jest mocny sygnał pozwalający rozróżnić OK i NOK. Sposób zbierania danych wymaga jednak korekty przed oceną skuteczności na nowych płytkach.**
MFCC z prostym klasyfikatorem wystarcza do pierwszej separacji. Transformer nie poprawił tego wyniku.
Jednocześnie sam format i parametry nagrania pozwalają zgadywać klasę. Nie umiemy jeszcze oddzielić wpływu pęknięcia od wpływu sposobu nagrywania.

## Co przeanalizowaliśmy

- {facts["audio_files"]} plików, {facts["duration_s"] / 60:.2f} minuty dźwięku, 48 kHz. Nowa paczka: 21 WAV-ów.
- 3613 kandydatów na uderzenia po filtrach jakości; to nie 3613 niezależnych przykładów materiału.
- Główny test: 27 nagrań, 2621 kandydatów, **2 płytki NOK i roboczo 3 płytki OK**.
- Folder główny OK, Part 2 i Part 3 traktujemy jako różne sztuki na podstawie założenia zaakceptowanego przez użytkownika. Potrzebne potwierdzenie klienta.
- OLD ma nieznane powiązania fizycznych płytek; sprawdzono go dodatkowo, poza głównym wynikiem. Bitten ma inny typ uszkodzenia i nie jest klasą pęknięć.
- G11, G12 i G16 trwają 0,47–1,20 s i nie weszły do modeli. Detekcje pozostają w audycie.

## Czy jakość pomiarów wystarcza

**Do eksploracji: tak. Do wiarygodnego pomiaru tłumienia i deklaracji skuteczności na nowych sztukach: jeszcze nie.**

1. **Wyciszanie sygnału.** W 17 z 45 plików występuje ciąg dokładnych zer dłuższy niż 100 ms, w tym w 10 nowych WAV-ach OK. W G13 trwa aż około 1,40 s. Nie jest to tylko artefakt stratnego AAC. Podejrzenie obejmuje bramkowanie, odszumianie, edycję lub przerwy toru. Nie rozpoznajemy konkretnego mechanizmu i nie dowodzimy utraty początku uderzenia. Cichy ogon jest szczególnie zagrożony.
2. **Zbyt szybka seria do badania pełnego wybrzmienia.** Typowy odstęp to około 0,28 s. Dlatego używamy wspólnego okna 20 ms przed + 150 ms po początku. Wzrost liczby uderzeń nie zastępuje nagrania dłuższego ogona. Na wykresie niebiesko-zielone pola oznaczają okno cech, a nie zmierzony fizyczny czas tłumienia.
3. **Warunki zmieniają się razem z klasą.** Nowe OK są w PCM WAV; 7 z 9 głównych nagrań NOK jest w AAC. Zmieniają się również podpory, narzędzie, kąt, długość nagrania i tło. Model może korzystać z tych różnic.
4. **Brakuje identyfikatorów i pełnego protokołu.** Potrzebne: ID płytki, typ/wymiary, temperatura, miejsce uderzenia, narzędzie, podparcie, odległość i orientacja mikrofonu, gain, aplikacja i ustawienia przetwarzania. OK również różnią się widmem; sprawność nie oznacza identycznego dźwięku.
5. **Tło jest za krótkie i częściowo wyciszone.** Otrzymaliśmy około 31,6 s zamiast ~10 min przewidzianych w minimalnej propozycji.

Pozytywy: wspólne 48 kHz, brak próbek osiągających przyjęty próg przesterowania 0,999 i liczne czytelne impulsy. Kanały stereo są identyczne we wszystkich 45 plikach, więc analizujemy jeden. Brak twardego clippingu nie dowodzi braku limitera lub AGC.

## Jak sprawdziliśmy predykcję

Z każdego użytecznego nagrania wybrano 10 uderzeń równomiernie w czasie, bez patrzenia na wynik modelu. Wszystkie reprezentacje porównujemy na tych samych zdarzeniach. Cechy klasyczne są dostępne także dla wszystkich kandydatów.

W każdej z 6 prób odkładamy **całą jedną płytkę NOK i całą jedną OK**. Trenujemy na pozostałych grupach. Skalowanie uczy się wyłącznie na treningu. Wagi równoważą klasy i grupy; długa seria nie dominuje uczenia. Nie stroiliśmy parametrów, progu ani wyboru cech pod ten test.

Wynik nagrania to mediana ocen 10 uderzeń. Tabela uśrednia oceny z prób, w których nagranie było odłożone; próg wynosi 0,5. To mały eksperyment eksploracyjny, nie niezależny końcowy test produkcyjny. Oceny modelu nie są skalibrowanym prawdopodobieństwem pęknięcia.

MFCC opisuje kształt widma, czyli w przybliżeniu barwę dźwięku. Physics to 15 cech opisujących częstotliwości i zanik. MFCC + physics łączy 26 + 15 = 41 cech w jednym wektorze, dla LR i XGB. Więcej cech nie gwarantuje poprawy: może również ułatwić uczenie różnic stanowiska. LR to prosty klasyfikator liniowy. XGB to zespół małych drzew. AST jest zamrożonym transformerem AudioSet, który tworzy 768 liczb opisujących dźwięk; dopiero LR lub XGB uczy się naszych etykiet. AST pracuje przy 16 kHz, traci pasmo powyżej 8 kHz i standardowo dopełnia krótkie uderzenia do długiego wejścia. Nie dostrajaliśmy transformera.

## Wyniki i ich granice

MFCC + LR: po uśrednieniu odłożonych predykcji 27/27 nagrań poprawnych, 9/9 nagrań NOK wykrytych. **W jednej pojedynczej próbie odrzucił jedno nagranie OK**; balanced accuracy poszczególnych prób wynosi {mainfolds.balanced_accuracy.min():.0%}–{mainfolds.balanced_accuracy.max():.0%}. Wyniku 100% z tabeli nie należy opisywać jako bezbłędności każdej próby.
AST + LR: 25/27 poprawnych, 8/9 NOK, jedno fałszywe odrzucenie OK. AST + XGB: 24/27. Połączenie MFCC + physics daje 27/27 dla LR i 26/27 dla XGB, tyle samo co samo MFCC. Prostsze MFCC pozostaje punktem odniesienia.

Po zebraniu ocen wszystkich nagrań danej grupy MFCC i AST poprawnie klasyfikują 5/5 zakładanych płytek. **To nadal tylko dwie uszkodzone sztuki.** Powtarzane podziały ani tysiące uderzeń nie zwiększają tej liczby. Nie podajemy sztucznie wąskich przedziałów ufności liczonych po uderzeniach.

Najważniejszy test kontrolny: reguła „AAC = NOK, WAV = OK” daje {codec["accuracy"]:.1%} accuracy bez słuchania uderzeń. Model QC bez informacji o kodeku, używający tylko długości, poziomu i ciszy, daje 88,9%. To nie dowodzi, że MFCC korzysta wyłącznie z tych skrótów, ale uniemożliwia przypisanie całego wyniku pęknięciom.

Po ograniczeniu do 4 podpór bez zmiany narzędzia MFCC i AST rozdzielają 10/10 nagrań. Jednak oba NOK w tym podzbiorze nadal są AAC, więc kontrola formatu też jest idealna. Samo dopasowanie podpór nie usuwa problemu.

Dwa nagrania NOK w WAV (R07, R08) również są rozpoznane przez MFCC, gdy cała płytka Bad_2 jest odłożona. To pozytywna poszlaka, że nie chodzi wyłącznie o rozróżnienie kodeka. Nadal pozostają inne różnice między sesjami i obiektami.

Dokładne przetasowanie etykiet między pięcioma grupami ma tylko 10 możliwych przydziałów dwóch NOK. Rzeczywisty przydział ma najwyższe AUC, ale rozdzielczość tego testu to p={p:.2f}. To nie jest mocny dowód statystyczny; wymienność grup dodatkowo ograniczają różnice zbierania danych.

W OLD, wyłączonym z uczenia, MFCC poprawnie rozpoznaje 6/8 plików, AST 7/8. Nie traktujemy tego jako niezależnego testu płytek: mogą to być nagrania tych samych sztuk.

## Szum i liczba uderzeń

Ściszenie o 12 dB nie zmienia wyników, zgodnie z normalizacją amplitudy. Po dodaniu aktywnego fragmentu nagranego tła do odłożonych uderzeń MFCC nadal wykrywa 9/9 nagrań NOK, ale fałszywie odrzuca 5/18 OK przy SNR 20 dB i 6/18 przy 10 dB. Niższy SNR oznacza mocniejsze tło. To kontrolowana domieszka do okien już wykrytych, nie test detektora na zaszumionej całej sesji. Test obejmuje teraz również AST + LR i AST + XGB oraz połączenie MFCC + physics. Tabela poniżej pokazuje wszystkie warianty. Dla każdego modelu używamy dokładnie tych samych zaszumionych uderzeń; modele uczą się wyłącznie na oryginałach. SNR odnosi się do mocy całego oryginalnego okna względem dodawanego tła, nie do nieznanego czystego dźwięku płytki.

**Połączenie cech nie poprawiło odporności w sposób spójny.** Przy 20 dB MFCC + physics + LR odrzuca 4/18 OK zamiast 5/18 dla MFCC + LR; przy mocniejszym tle 10 dB odrzuca już 10/18 zamiast 6/18. AST + LR odrzuca odpowiednio 4/18 i 8/18 OK. AST + XGB przy 10 dB odrzuca aż 17/18 OK. Wszystkie te warianty po domieszaniu tła wykrywają 9/9 NOK. Wysoki recall nie wystarcza, jeśli prawie wszystko zostaje uznane za wadliwe. AUC może pozostać wysokie, choć decyzje przy stałym progu 0.5 się pogarszają: tło przesuwa oceny ku NOK. Nie poprawiamy progu na tych samych testach.

Wykres 1/3/5/10 uderzeń pokazuje stabilność mediany ocen na tych samych odłożonych nagraniach. Pasmo oznacza rozrzut 100 losowań, nie przedział ufności skuteczności na nowych płytkach. Nie wyznaczamy jeszcze minimalnej liczby uderzeń do produkcji ani nie porównujemy „master tap” bez pozycji trafień.

## Co zmienić przy następnej paczce

**Najpierw krótki test ustawień.** Jedna OK i jedna NOK, ta sama geometria, narzędzie i gain. Dla każdej wariant obecny oraz z wyłączonymi ulepszeniami wejścia mikrofonowego, odszumianiem, AGC i usuwaniem echa. 10 s tła, 10 uderzeń z przerwami 2–3 s, 10 s tła. Zapisać bezpośrednio WAV PCM 48 kHz, najlepiej 24 bit, bez efektów. Nie konwertować starego AAC jako substytutu nowego pomiaru. Potwierdzić fizycznie wybrane wejście mikrofonowe.

**Następnie powtarzalna seria wszystkich dostępnych płytek.** Jeden ustalony schemat podpór i narzędzie; 5 oznaczonych miejsc po 2 uderzenia. Te same miejsca dla OK i NOK. Nagrywać kolejno OK, NOK, OK, NOK, aby zmiana tła w czasie nie pokrywała się z klasą. Powtórzyć po zdjęciu i ponownym ułożeniu płytek, zapisać session_id. Siła możliwie podobna, bez dociskania płyty podczas wybrzmienia. Zmiany narzędzia i podpór dopiero jako osobny, jawnie oznaczony eksperyment dla obu klas.

**Więcej różnych sztuk, nie setki tapów tej samej.** Zachować co najmniej 30 OK i 10 NOK jako cel danych ze slajdu minimalnego PoC. Potwierdzić etykiety i zdjęcia, oddzielić pęknięcia od wyszczerbień. Nagrać około 10 min nieprzetworzonego tła w typowych warunkach.

miniDSP wprost zaleca wyłączenie ulepszeń Windows dla UMIK-1. Mikrofon jest w propozycji przykładem; model faktycznie użyty wymaga potwierdzenia. Kalibracja charakterystyki mikrofonu i ręczny gain nie są tym samym co adaptacyjne odszumianie.

## Zakres i odtwarzanie

To realizuje etap QA, spektrogramów, klasycznych cech i prób rozróżniania z minimalnego PoC. AST jest dodatkiem porównawczym, nie zmianą zobowiązania klienta. Przejrzano wykresy reprezentatywnych fragmentów, nie odsłuchano ręcznie wszystkich 3613 kandydatów. Notebook zawiera odtwarzacz i przegląd do zatwierdzenia segmentacji.

Notebooki: `notebooks/client/02_jakosc_obu_paczek.ipynb` i `03_modele_obu_paczek.ipynb`. Wybierz kernel `.venv` i Run All. Pierwszy sprawdza dane i pozwala klikać nagrania. Drugi przelicza modele, podziały, testy kontrolne, szum i liczbę uderzeń. Wagi AST są już lokalnie dostępne na tym Macu; nie wysyłamy audio do usługi zewnętrznej.

Źródła: slajd 1 `docs/Hitachi PoC proposals 2.pptx`, roboczy `docs/Hitachi DQC - Methods Guideline (PoC Sprint).docx`; producent miniDSP; model MIT AST; dokumentacja walidacji grupowej scikit-learn.
"""
    (OUT / "RAPORT.md").write_text(md)
    # Record a copy in tracked docs, without placing private raw audio into the HTML.
    (ROOT / "docs/CLIENT_COMBINED_REPORT.md").write_text(md)
    import mistune

    body = mistune.html(md)
    body = body.replace(
        "<h2>Wyniki i ich granice</h2>", "<h2>Wyniki i ich granice</h2>" + table(comp) + picture("models.png")
    )
    body = body.replace(
        "<h2>Jak sprawdziliśmy predykcję</h2>",
        picture("spectra.png")
        + picture("pca.png")
        + "<p>Kolory: czerwony Bad_1, pomarańczowy Bad_2, turkusowy OK_1, niebieski OK_2, zielony OK_3. Trójkąt: jedna podpora, koło: pozostałe. PCA dopasowano do całości wyłącznie do wizualizacji.</p><h2>Jak sprawdziliśmy predykcję</h2>",
    )
    body = body.replace(
        "<h2>Co zmienić przy następnej paczce</h2>",
        table(noise) + picture("tap_count.png") + "<h2>Co zmienić przy następnej paczce</h2>",
    )
    body = body.replace(
        "<h2>Jak sprawdziliśmy predykcję</h2>",
        picture("quiet_intervals.png") + "<h2>Jak sprawdziliśmy predykcję</h2>",
    )
    # Interactive offline gallery: original audio + overview + native QC. No server needed.
    gallery = []
    for r in m.itertuples():
        path = OUT / "figures" / f"{r.recording_id}_overview.png"
        if not path.exists():
            y, sr = sf.read(OUT / "decoded" / f"{r.recording_id}.wav")
            step = max(1, len(y) // 10000)
            fig, ax = plt.subplots(figsize=(11, 2.5), constrained_layout=True)
            ax.plot(np.arange(0, len(y), step) / sr, y[::step], lw=0.6, color="#167f86")
            ax.set(title=r.recording_id, xlabel="Czas [s]", ylabel="Amplituda")
            fig.savefig(path, dpi=120)
            plt.close(fig)
        gallery.append(
            dict(
                id=r.recording_id,
                name=r.filename,
                audio=quote(os.path.relpath(ROOT / r.path, OUT)),
                image="figures/" + path.name,
                seconds=round(r.duration_s, 2),
                accepted=int(r.n_accepted),
                zero=round(r.longest_zero_s, 3),
                group=str(r.group),
                codec=r.codec,
            )
        )
    gallery_html = """<section id="gallery"><h2>Odsłuch i przegląd wszystkich nagrań</h2><select id="recording" aria-label="Wybierz nagranie"></select><p id="details"></p><audio id="audio" controls preload="metadata"></audio><img id="wave" class="chart" alt="Przebieg wybranego nagrania"><p>Przebieg jest uproszczonym podglądem. Dokładne okna i spektrogram znajdziesz w notebooku jakości.</p></section>"""
    source_links = """<p>Źródła zewnętrzne: <a href="https://www.minidsp.com/community/threads/umik-1-windows-audio-enhancement.19359/">miniDSP o ulepszeniach</a> · <a href="https://www.minidsp.com/products/acoustic-measurement/umik-1">UMIK-1</a> · <a href="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593">AST</a> · <a href="https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data">Walidacja grupowa</a></p>"""
    page = (
        """<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ocena nagrań OK i NOK</title><style>
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f3ef;color:#202d36;font:17px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:40px auto;background:white;padding:40px 56px;border-radius:12px}h1{font-size:38px;line-height:1.15;max-width:800px}h2{margin-top:48px;font-size:25px;border-top:1px solid #e2e6e7;padding-top:22px}p,li{max-width:980px}strong{color:#174c54}a{color:#126b7b}.chart{display:block;width:100%;height:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto}th,td{padding:9px 11px;text-align:left;border-bottom:1px solid #e2e6e7;white-space:nowrap}th{background:#e8f1f1}select{font:inherit;width:100%;padding:10px}audio{width:100%}nav{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:30px;font-size:14px}@media(max-width:750px){main{margin:0;padding:22px}h1{font-size:30px}}
    </style><main><nav><a href="#gallery">Odsłuch</a><a href="RAPORT.md">Raport tekstowy</a><a href="02_jakosc_obu_paczek.html">Notebook jakości</a><a href="03_modele_obu_paczek.html">Notebook modeli</a></nav>"""
        + body
        + gallery_html
        + source_links
        + """</main><script>
    const data="""
        + json.dumps(gallery, ensure_ascii=False).replace("</", "<\\/")
        + """;const sel=document.getElementById('recording');data.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id+' · '+r.name;sel.append(o)});function show(){const r=data[+sel.value];document.getElementById('audio').src=r.audio;document.getElementById('wave').src=r.image;document.getElementById('details').textContent=r.codec+' · '+r.seconds+' s · '+r.accepted+' kandydatów użytych · najdłuższy ciąg zer '+r.zero+' s · grupa '+r.group;}sel.addEventListener('change',show);show();</script></html>"""
    )
    if (OUT / "explanations/START.html").exists():
        page = page.replace("</nav>", '<a href="explanations/START.html">Cechy i tło</a></nav>', 1)
    (OUT / "START.html").write_text(page)


SETUP = """from pathlib import Path
import sys, json, io, contextlib
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "config/client_archive_manifest.json").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Otwórz folder ceramic-ping-lab w VS Code.")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
from pinglab.clientstudy import prepare, load, embeddings, recording_widget, OUT_REL
OUT = ROOT / OUT_REL
pd.set_option("display.max_colwidth", 65)
%matplotlib inline
print("Python:", sys.executable)
print("Wyniki:", OUT)
"""


def notebook(name, cells):
    n = nbf.v4.new_notebook(
        cells=[nbf.v4.new_markdown_cell(s) if kind == "m" else nbf.v4.new_code_cell(s) for kind, s in cells]
    )
    n.metadata.kernelspec = dict(
        display_name="Ping Lab (Python)", language="python", name="pinglab"
    )
    n.metadata.language_info = dict(name="python", version="3.12.14")
    path = ROOT / "notebooks/client" / name
    nbf.validate(n)
    nbf.write(n, path)


def marker(kind):
    return f'''from datetime import datetime, timezone
verification = dict(notebook="{kind}", finished_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable)
(OUT / "last_run_{kind}.json").write_text(json.dumps(verification, indent=2))
print("GOTOWE — {kind}: wszystkie komórki zakończone.")'''


def notebooks():
    notebook(
        "02_jakosc_obu_paczek.ipynb",
        [
            (
                "m",
                """# Czy te nagrania nadają się do analizy?
**Zacznij tutaj.** Wybierz kernel `.venv` i kliknij **Run All**. Oryginały są w `assets/1-batch-ping` i `assets/2-batch-ok`.
Notebook sprawdza pliki, wykrywa uderzenia i daje odsłuch z wykresami. Nie zmienia źródłowego audio.

Wniosek: impulsy nadają się do eksploracji. Trzeba wyjaśnić wyciszanie tła, ujednolicić zapis i wydłużyć przerwy przed oceną pełnego wybrzmienia. Dwie NOK i roboczo trzy OK to nadal pięć obiektów.""",
            ),
            ("c", SETUP),
            (
                "m",
                "## 1. Wczytanie obu paczek\nCache przyspiesza kolejne uruchomienia; jego klucz zależy od SHA256 audio i wersji ekstrakcji. Zmień `FORCE = True`, aby przeliczyć cechy od początku.",
            ),
            (
                "c",
                """FORCE = False
with contextlib.redirect_stdout(io.StringIO()):
    recordings, candidates, features = prepare(ROOT, force=FORCE)
from pinglab.clientdiagnostics import figures
facts = figures(ROOT)
display(pd.DataFrame([facts]).T.rename(columns={0:"Wartość"}))
display(recordings.groupby(["batch","scope"]).agg(pliki=("recording_id","size"), sekundy=("duration_s","sum"), kandydaci=("n_accepted","sum")).round(2))""",
            ),
            (
                "m",
                "## 2. Co wymaga wyjaśnienia\n**Ciąg zer >100 ms** to powód do sprawdzenia toru pomiarowego. Pojedyncze zera w WAV 16-bit mogą wynikać z kwantyzacji. Nie nazywamy każdego zera odszumianiem. Trzy pliki krótsze niż 2 s zachowujemy w spisie, lecz nie uczymy na nich modeli.",
            ),
            (
                "c",
                """display(recordings.loc[(recordings.longest_zero_s > .1) | (recordings.duration_s < 2),
    ["recording_id","filename","duration_s","longest_zero_s","below_minus100_fraction","n_accepted"]].round(3))
display(Image(filename=str(OUT / "figures/quiet_intervals.png")))""",
            ),
            (
                "m",
                "## 3. Klikalny odsłuch, segmentacja i spektrogram\nWybierz plik, początek i długość fragmentu. Zielone pola to **stałe okna cech 20 ms przed +150 ms po początku**. Złote linie oznaczają wykrycia odrzucone przez reguły jakości. To nie pomiar fizycznego końca wybrzmienia. Audio zachowuje poziom źródłowy; może być ciche.",
            ),
            ("c", """controls = recording_widget(ROOT)"""),
            (
                "m",
                "## 4. Czy detektor widzi pojedyncze uderzenia?\nNiżej te same reguły dla kilku OK i NOK oraz krótkiego G11. Detektor nie wymusza zadanej liczby trafień. Tabela pokazuje, jak liczba kandydatów zależy od minimalnego odstępu. Manualne zatwierdzenie wszystkich trafień pozostaje do wykonania.",
            ),
            (
                "c",
                """display(Image(filename=str(OUT / "figures/detector_review.png")))
display(recordings.loc[recordings.scope.eq("core"),["recording_id","n_candidates","n_gap250","n_gap400","n_accepted","median_gap_s"]].round(3))
display(candidates[candidates.accepted.eq(False)].rejection.value_counts().rename("Liczba"))""",
            ),
            (
                "m",
                "## 5. Co słychać w widmie i co zmienia stanowisko\nPik to mocno obecna częstotliwość drgań. Krzywa pokazuje rozkład energii po częstotliwościach. Wysokość znormalizowano, aby porównać kształt. Prawa część ogranicza porównanie do 4 podpór bez zmiany narzędzia. Różnica widma sama nie dowodzi, że jej przyczyną jest pęknięcie.",
            ),
            (
                "c",
                """display(Image(filename=str(OUT / "figures/spectra.png")))
display(Image(filename=str(OUT / "figures/conditions.png")))""",
            ),
            (
                "m",
                "## 6. Co poprosić operatora\n1. Krótki test z ulepszeniami wejścia wyłączonymi: 1 OK i 1 NOK, 10 s tła, 10 uderzeń co 2–3 s, 10 s tła.\n2. Bezpośredni WAV PCM 48 kHz, ręczny stały gain, potwierdzone wejście mikrofonowe.\n3. Jedno narzędzie i schemat podpór, 5 oznaczonych miejsc po 2 uderzenia, identycznie dla obu klas.\n4. ID płytki i sesji, zdjęcie i rodzaj wady, typ płyty, temperatura, miejsce uderzenia oraz geometria mikrofonu.\n5. Kolejne różne płytki oraz ~10 min tła. Potem sprawdzimy nową sesję bez zmiany modelu.\n\nPrzejdź do **03_modele_obu_paczek.ipynb**. Raport zbiorczy: `reports/runs/client-combined-20260929/START.html`.",
            ),
            ("c", marker("quality")),
        ],
    )
    notebook(
        "03_modele_obu_paczek.ipynb",
        [
            (
                "m",
                """# Czy już umiemy odróżnić OK od NOK?
**Run All** przelicza modele na tym samym podziale po całych płytkach. Wybierz `.venv`.
Mamy 2 NOK i roboczo 3 OK. Foldery OK są różnymi płytkami **na podstawie przyjętego założenia**, nie potwierdzonych ID klienta.
Wynik jest obiecującą poszlaką. Sposób zapisu jest powiązany z klasą, więc nie wolno przenosić accuracy z tej paczki na fabrykę.""",
            ),
            ("c", SETUP),
            (
                "m",
                "## 1. Te same uderzenia dla każdej metody\nWybieramy 10 trafień równomiernie w czasie z każdego użytecznego pliku. AST tworzy embedding, czyli wektor 768 liczb. Wagi są zamrożone i już zapisane lokalnie. Nie uczymy transformera. AST pracuje przy 16 kHz, więc nie widzi pasma >8 kHz. 170 ms dopełnia standardowy procesor modelu.",
            ),
            (
                "c",
                """with contextlib.redirect_stdout(io.StringIO()):
    recordings, candidates, features = prepare(ROOT)
    representations = embeddings(ROOT)
display(representations.groupby(["scope","group"]).agg(nagrania=("recording_id","nunique"),uderzenia=("event_id","size")))
print("Physics: 15 cech; MFCC: 26; MFCC + physics: 41; mel: 40; AST: 768.")""",
            ),
            (
                "m",
                "## 2. Mapa podobieństwa\nKażdy punkt to nagranie. Kolory: Bad_1 czerwony, Bad_2 pomarańczowy, OK_1 turkusowy, OK_2 niebieski, OK_3 zielony. Trójkąt: jedna podpora, koło: pozostałe. PCA streszcza wektor w 2 liczbach i służy tylko do oglądania. Model używa całego wektora; ta mapa nie mierzy accuracy.",
            ),
            (
                "c",
                """from pinglab.clientdiagnostics import figures
figures(ROOT)
display(Image(filename=str(OUT / "figures/pca.png")))""",
            ),
            (
                "m",
                "## 3. Trening i uczciwszy podział\n6 prób: odkładamy całą jedną NOK i jedną OK. Żaden tap tej samej grupy nie przechodzi między treningiem i testem. Skaler uczy się wyłącznie na treningu. Stałe LR C=0.1 i płytki XGBoost, bez szukania najlepszych parametrów. Klasy i grupy mają równe wagi. Powtarzane próby nie są nowymi płytkami.",
            ),
            (
                "c",
                """from pinglab.clientmodels import run
with contextlib.redirect_stdout(io.StringIO()):
    scores = run(ROOT)
audit = json.loads((OUT / "split_audit.json").read_text())
display(pd.DataFrame(audit)[["fold","train_groups","test_groups","train_taps","test_taps"]])""",
            ),
            (
                "m",
                "## 4. Porównanie modeli\n**MFCC + physics** to wspólny wektor 41 cech: 26 MFCC i 15 fizycznych. Wcześniejsze rozdzielenie było punktem odniesienia, nie wynikiem testu korelacji. XGBoost potrafi ignorować część cech, ale dodatkowe cechy mogą też wzmacniać przypadkowe zależności przy zaledwie pięciu płytkach. Sprawdzamy połączenie na identycznych podziałach, bez dostrajania parametrów do wyniku.\n\n**Accuracy**: ile decyzji jest poprawnych. **Recall NOK**: ile wadliwych wykryliśmy. **Fałszywe odrzucenia OK**: ile dobrych odrzuciliśmy. **Balanced accuracy** nadaje obu klasom równą wagę. **AUC** ocenia uporządkowanie wyników bez konkretnego progu. Próg 0.5 jest stały, a oceny nie są skalibrowanym prawdopodobieństwem wady.\n\nTabela nagrań uśrednia przewidywania ze wszystkich prób, w których dany plik był odłożony. W jednej próbie MFCC odrzuca jedno OK, choć po uśrednieniu tabela pokazuje 27/27. Poniżej są również osobne próby i tylko pięć zakładanych płytek.",
            ),
            (
                "c",
                """cols = ["model","n","accuracy","balanced_accuracy","recall_nok","false_reject_ok","auc","tp","fn","tn","fp"]
display(scores[scores.level.eq("recording")][cols].round(3))
display(scores[scores.level.eq("assumed_plate")][cols].round(3))
folds = pd.read_csv(OUT / "fold_metrics.csv")
display(folds[folds.model.isin(["MFCC + LR","MFCC + physics + LR","MFCC + physics + XGB","AST + LR"])].round(3))""",
            ),
            (
                "m",
                "## 5. Najważniejsza kontrola: czy rozpoznajemy sposób nagrywania?\nQC-only używa długości, poziomu, ciszy i kodeka. QC-no-codec usuwa nawet informację o formacie. Ich wysokie wyniki ostrzegają przed myleniem właściwości płytek ze sposobem zbierania. Reguła AAC=NOK jest celowo prymitywna i nie analizuje widma.\n\nPodzbiór 4 podpór usuwa część różnic stanowiska, ale pozostawia AAC po stronie NOK i WAV po stronie OK. Dwa NOK w WAV, R07/R08, też są rozpoznane przez MFCC; to poszlaka szerszego sygnału, nie rozwiązanie wszystkich zakłóceń.",
            ),
            (
                "c",
                """display(pd.DataFrame([json.loads((OUT / "codec_control.json").read_text())]).round(3))
display(pd.read_csv(OUT / "matched_support_metrics.csv").query("level == 'recording'").round(3))
file_predictions = pd.read_csv(OUT / "oof_recordings.csv")
display(file_predictions.query("model == 'MFCC + LR' and recording_id in ['R07', 'R08']"))
perms = pd.read_csv(OUT / "group_label_permutations.csv")
print("Tylko", len(perms), "możliwych przydziałów etykiet; p dla AUC >= rzeczywistego:", (perms.auc >= 1-1e-12).mean())
display(pd.read_csv(OUT / "old_metrics_diagnostic.csv").round(3))
print("OLD: diagnostyka innych plików, nie potwierdzony test niezależnych płytek.")""",
            ),
            (
                "m",
                "## 6. Odporność na tło: także AST i MFCC + physics\nUczymy na oryginałach, zmieniamy tylko odłożone okna. Wszystkie osiem modeli dostaje te same zmienione sygnały. Używamy nagranego tła z sekund 19–30, dodanego matematycznie do uderzeń. Dobieramy jego amplitudę: RMS(tła) = RMS(oryginału) / 10^(SNR/20). Przy 20 dB tło ma 1/10 amplitudy RMS oryginału, przy 10 dB około 1/3,16. To SNR dodanej domieszki względem całego okna, które już zawiera własne tło.\n\nAST ponownie oblicza embedding zaszumionego audio; nie dodajemy szumu do wektorów. Pierwsze uruchomienie jest dłuższe, kolejne korzystają z lokalnego cache. Test obejmuje klasyfikację, bez ponownej detekcji uderzeń. Ściszenie sprawdza normalizację amplitudy.\n\n**Jak czytać wynik:** przy SNR 10 dB MFCC + LR odrzuca 6/18 dobrych, połączenie MFCC + physics + LR 10/18, AST + LR 8/18, a AST + XGB 17/18. Wszystkie wykrywają 9/9 NOK. Sam recall wygląda więc dobrze, ale odrzucanie dobrych rośnie. Przy 20 dB połączenie LR ma o jeden fałszywy alarm mniej niż samo MFCC; nie ma spójnej przewagi przy obu poziomach tła. Wysokie AUC i gorsze decyzje wskazują na przesunięcie ocen względem stałego progu 0.5. Nie dostrajamy go do tego testu.",
            ),
            (
                "c",
                """from pinglab.clientdiagnostics import perturbations
noise_results = perturbations(ROOT)
display(noise_results[["model","variant","accuracy","recall_nok","false_reject_ok","auc","tp","fn","fp"]].round(3))
display(noise_results.pivot(index="model", columns="variant", values="balanced_accuracy").round(3))
snr_audit = pd.read_csv(OUT / "noise_snr_audit.csv")
print("Maksymalny błąd ustawienia SNR [dB]:", (snr_audit.measured_snr_db-snr_audit.requested_snr_db).abs().max())""",
            ),
            (
                "m",
                "## 7. Ile uderzeń do decyzji?\nLosujemy 1/3/5/10 z tych samych odłożonych uderzeń, 100 razy. Pokazujemy medianę ocen. Pasmo to zmienność losowania, nie pewność statystyczna na nowych płytkach. Bez pozycji trafień nie ocenimy, czy jedno konkretne uderzenie centralne wystarczy.",
            ),
            (
                "c",
                """import importlib.util
spec = importlib.util.spec_from_file_location("combined_review", ROOT / "scripts/build_combined_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
review.performance_figures()
display(Image(filename=str(OUT / "figures/tap_count.png")))
taps = pd.read_csv(OUT / "tap_count_resampling.csv")
display(taps.groupby(["model","k"])[["balanced_accuracy","recall_nok","false_reject_ok"]].mean().round(3))""",
            ),
            (
                "m",
                "## 8. Decyzja po tej rundzie\n**Kontynuować PoC.** MFCC jest dobrym prostym punktem odniesienia, AST jest dodatkową kontrolą. Największy zysk teraz da kontrolowana nowa sesja tych samych płytek z jednakowym zapisem oraz więcej fizycznych sztuk, szczególnie NOK. Nie dostrajamy progu do uzyskania zerowej liczby pomyłek na tych samych danych.\n\nPełny opis, zalecenia i atlas są w START.html. Surowe audio i wyniki pozostają lokalnie.",
            ),
            (
                "c",
                """review.write_report()
print("Raport:", OUT / "START.html")
"""
                + marker("models"),
            ),
        ],
    )


if __name__ == "__main__":
    performance_figures()
    write_report()
    notebooks()
    print(OUT / "START.html")

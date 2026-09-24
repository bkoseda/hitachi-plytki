import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .plots import embedding_map, prediction_plot

STYLE = """body{margin:0;background:#f6f6f1;color:#203039;font:16px/1.55 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:36px 24px}h1{font-size:32px;line-height:1.2}h2{font-size:22px}
.note{background:#fff0d8;border-left:4px solid #b56d20;padding:14px 18px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:9px;border-bottom:1px solid #d7dfdc}th{background:#eaf0ed}.scroll{overflow:auto}
.stats{display:flex;gap:30px;flex-wrap:wrap}.stats b{font-size:26px;display:block}a{color:#117c83}code{background:#e6eeeb;padding:2px 5px}
"""


def write_report(features, qc, path, evaluation=None, title="Raport analizy", source_kind="client"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    e = html.escape
    parts = [
        f'<!doctype html><html lang="pl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{e(title)}</title><style>{STYLE}</style><main>',
        f"<p>PING LAB / {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p><h1>{e(title)}</h1>",
    ]
    if source_kind == "synthetic":
        parts.append(
            '<p class="note"><strong>DANE SYNTETYCZNE.</strong> Wyniki sprawdzają działanie narzędzi. Nie są dowodem skuteczności dla ceramiki klienta.</p>'
        )
    elif source_kind == "external":
        parts.append(
            '<p class="note">Publiczne dane z innej domeny. Brak podstaw do wniosków o wadach płytek klienta.</p>'
        )
    else:
        parts.append(
            '<p class="note">Wynik eksploracyjny. Ocena jakości i segmentacji wymaga odsłuchu; progi nie są kryteriami odbiorowymi.</p>'
        )
    nplates = features.plate_id.nunique() if not features.empty else 0
    parts.append(
        f'<div class="stats"><div><b>{len(qc)}</b>plików</div><div><b>{nplates}</b>płytek z uderzeniami</div><div><b>{len(features)}</b>uderzeń</div></div>'
    )
    unreviewed = int((~qc.reviewed.astype(bool) & (qc.label != "background")).sum())
    parts.append(
        f"<p><strong>Pliki bez ręcznie zatwierdzonych czasów: {unreviewed}.</strong> Sprawdź każdy przed interpretacją modelu.</p>"
    )
    visible_qc = qc[
        ["path", "label", "n_hits", "reviewed", "peak_dbfs", "clipping_fraction", "warnings"]
    ].copy()
    visible_qc["clipping_fraction"] *= 100
    visible_qc["reviewed"] = visible_qc["reviewed"].map({True: "tak", False: "nie"})
    visible_qc = visible_qc.rename(
        columns={
            "path": "Plik",
            "label": "Klasa",
            "n_hits": "Uderzenia",
            "reviewed": "Sprawdzone",
            "peak_dbfs": "Szczyt [dBFS]",
            "clipping_fraction": "Pełna skala [%]",
            "warnings": "Uwagi",
        }
    )
    parts.append(
        '<details><summary>Jakość nagrań — rozwiń tabelę</summary><div class="scroll">'
        + visible_qc.to_html(index=False, escape=True, classes="qc", float_format=lambda v: f"{v:.3f}")
        + "</div></details>"
    )
    include = True
    if nplates >= 3:
        fig, _ = embedding_map(features, evaluation[0]["family"] if evaluation else "physics")
        parts.append(
            "<h2>Czy reprezentacje są podobne?</h2><p>PCA służy do oglądania danych. Separacja na tym wykresie nie jest wynikiem testu.</p>"
        )
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include))
        include = False
    if evaluation:
        metrics, predictions, audit = evaluation
        parts.append("<h2>Ocena na płytkach poza treningiem</h2>")
        parts.append(
            "<p>Mediana cech uderzeń na płytkę; stałe C=0.1, próg=0.5. Może ukryć lokalny defekt słyszalny tylko w jednym miejscu.</p>"
        )
        parts.append(
            f"<p>Reprezentacja: <strong>{e(metrics['family'])}</strong> · podział: {e(metrics['split'])}.</p>"
        )
        parts.append(
            '<div class="stats">'
            + "".join(
                f"<div><b>{metrics[k]:.1%}</b>{name}</div>"
                for k, name in [
                    ("balanced_accuracy", "balanced accuracy"),
                    ("recall_nok", "wykryte NOK"),
                    ("false_alarm_rate", "fałszywe alarmy OK"),
                ]
            )
            + "</div>"
        )
        parts.append("<p>Macierz pomyłek: wiersze rzeczywiste OK/NOK, kolumny przewidziane OK/NOK.</p>")
        parts.append("<pre>" + e(json.dumps(metrics["confusion_matrix"])) + "</pre>")
        if "recall_nok_wilson_95" in metrics:
            lo, hi = metrics["recall_nok_wilson_95"]
            flo, fhi = metrics["false_alarm_wilson_95"]
            parts.append(
                f"<p>Przedziały Wilsona 95%: recall NOK {lo:.1%}–{hi:.1%}; "
                f"fałszywe alarmy {flo:.1%}–{fhi:.1%}. " + e(metrics["wilson_note"]) + "</p>"
            )
        parts.append("<p>" + e(metrics["warning"]) + "</p>")
        if "balanced_accuracy_bootstrap_95" in metrics:
            lo, hi = metrics["balanced_accuracy_bootstrap_95"]
            parts.append(
                f"<p>Przedział bootstrap 95%: {lo:.1%}–{hi:.1%}. " + e(metrics["interval_note"]) + "</p>"
            )
            if hi - lo < 1e-10:
                parts.append(
                    "<p>Brak rozrzutu bootstrap przy tych prognozach nie oznacza braku niepewności; sprawdź liczebności i przedziały Wilsona.</p>"
                )
        parts.append(prediction_plot(predictions).to_html(full_html=False, include_plotlyjs=include))
        parts.append(
            '<div class="scroll">'
            + predictions.to_html(index=False, escape=True, float_format=lambda v: f"{v:.3f}")
            + "</div>"
        )
        (path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        (path.parent / "folds.json").write_text(json.dumps(audit, indent=2))
        predictions.to_csv(path.parent / "predictions.csv", index=False)
    else:
        parts.append(
            "<h2>Następny krok</h2><p>Porównaj odsłuch, widmo i wybrzmiewanie. Do oceny modelu potrzeba wielu różnych płytek obu klas; dwie próbki nie wystarczą.</p>"
        )
    parts.append(
        "<h2>Co przekazać po pierwszych plikach</h2><p>Jakość nagrania · widoczne uderzenia · słyszalna różnica · zmiana protokołu · czego brakuje. Nie podawaj skuteczności z porównania jednej pary.</p></main></html>"
    )
    path.write_text("\n".join(parts), encoding="utf-8")
    return path

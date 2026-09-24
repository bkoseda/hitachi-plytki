"""Notebook helpers: explicit upload and review, no automatic inference on file change."""

import hashlib
import json
from pathlib import Path

import ipywidgets as W
import numpy as np
from IPython.display import clear_output, display

from .audio import Config, cut_hits, detect_hits, listen, load_audio
from .dataset import read_manifest
from .plots import inspection


def upload_box(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    upload = W.FileUpload(accept=".wav,.mp3,.flac,.m4a,.aac,.ogg,.aiff,.aif,.wma,.webm", multiple=True)
    button = W.Button(description="Zapisz wybrane pliki", button_style="primary")
    output = W.Output()

    def save(_):
        with output:
            clear_output()
            for item in upload.value:
                content = bytes(item["content"])
                name = Path(item["name"]).name
                file = destination / name
                if file.exists() and file.read_bytes() != content:
                    file = destination / f"{file.stem}_{hashlib.sha256(content).hexdigest()[:8]}{file.suffix}"
                file.write_bytes(content)
                print(file)

    button.on_click(save)
    display(
        W.VBox(
            [
                W.HTML("<b>Pliki pozostają lokalnie. Po zapisie skopiuj ścieżki do konfiguracji.</b>"),
                upload,
                button,
                output,
            ]
        )
    )
    return upload


def review_manifest(manifest, annotation_path, config=None):
    config = config or Config()
    frame = read_manifest(manifest)
    frame = frame[frame.label != "background"]
    if frame.empty:
        raise ValueError("Brak nagrań płytek.")
    annotation_path = Path(annotation_path)
    annotations = json.loads(annotation_path.read_text()) if annotation_path.exists() else {}
    select = W.Dropdown(options=frame.path.tolist(), description="Nagranie:", layout=W.Layout(width="95%"))
    sensitivity = W.FloatSlider(
        value=config.sensitivity,
        min=2,
        max=15,
        step=0.5,
        description="Próg detekcji:",
        style={"description_width": "initial"},
        layout=W.Layout(width="400px", max_width="95%"),
    )
    times = W.Textarea(description="Czasy [s]:", layout=W.Layout(width="95%", height="70px"))
    inspect = W.Button(description="Wczytaj / wykryj")
    preview = W.Button(description="Podgląd czasów", disabled=True)
    save = W.Button(description="Zatwierdź i zapisz", button_style="success", disabled=True)
    output = W.Output()
    state = {}

    def reset_selection(_):
        state.clear()
        times.value = ""
        preview.disabled = save.disabled = True
        with output:
            clear_output()
            print("Nowy plik: kliknij Wczytaj / wykryj przed edycją czasów.")

    select.observe(reset_selection, names="value")

    def show(use_manual=False):
        if use_manual and state.get("path") != select.value:
            return
        with output:
            clear_output(wait=True)
            try:
                row = frame[frame.path == select.value].iloc[0]
                audio = load_audio(
                    row.resolved_path, config, channel=int(row.channel) if row.channel else None
                )
                if use_manual:
                    onsets = np.array([float(s.strip()) for s in times.value.split(",") if s.strip()])
                else:
                    from dataclasses import replace

                    onsets = np.asarray(
                        annotations.get(
                            select.value,
                            detect_hits(audio.y, audio.sr, replace(config, sensitivity=sensitivity.value)),
                        )
                    )
                    times.value = ", ".join(f"{v:.3f}" for v in onsets)
                segments = cut_hits(audio.y, audio.sr, onsets, config)
                state.update(path=select.value, onsets=onsets)
                preview.disabled = save.disabled = False
                print(
                    f"Szczyt: {audio.qc['peak_dbfs']:.1f} dBFS · "
                    f"blisko pełnej skali: {100 * audio.qc['clipping_fraction']:.3f}% · "
                    f"kanał: {audio.selected_channel}"
                )
                if audio.qc["warnings"]:
                    print(" | ".join(audio.qc["warnings"]))
                display(listen(audio.y, audio.sr, normalize=False))
                display(inspection(audio, onsets))
                print(f"{len(segments)} uderzeń. Usuń mowę, stuknięcia w stół i fałszywe podziały.")
                for i, segment in enumerate(segments):
                    print(
                        f"Uderzenie {i + 1} · {segment['start_s']:.3f}–{segment['end_s']:.3f} s"
                        + (" · skrócony ogon" if segment["truncated"] else "")
                    )
                    display(listen(segment["y"], audio.sr, normalize=True))
            except Exception as ex:
                state.clear()
                preview.disabled = save.disabled = True
                print(f"Błąd: {ex}")

    def commit(_):
        show(True)
        if state.get("path") == select.value:
            annotations[select.value] = state["onsets"].tolist()
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2))
            with output:
                print(f"Zapisano: {annotation_path}")

    inspect.on_click(lambda _: show(False))
    preview.on_click(lambda _: show(True))
    save.on_click(commit)
    display(W.VBox([select, sensitivity, W.HBox([inspect, preview, save]), times, output]))
    return dict(select=select, times=times, output=output, inspect=inspect, preview=preview, save=save)

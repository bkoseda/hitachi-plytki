import numpy as np

from .features import extract_features


def mix_noise(y, noise, snr_db, seed=42):
    y = np.asarray(y, float)
    noise = np.asarray(noise, float)
    if len(noise) == 0 or not np.isfinite(noise).all():
        raise ValueError("Brak poprawnego tła.")
    rng = np.random.default_rng(seed)
    noise = np.tile(noise, int(np.ceil((len(y) + len(noise)) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(y) + 1))
    noise = noise[start : start + len(y)]
    noise = noise - noise.mean()
    n_rms = np.sqrt(np.mean(noise**2))
    if n_rms < 1e-10:
        raise ValueError("Tło jest ciszą.")
    scale = np.sqrt(np.mean(y**2)) / (10 ** (snr_db / 20) * n_rms)
    return (y + noise * scale).astype(np.float32)


def sensitivity_table(y, sr, noise=None, offset=0.025, fmax=20000):
    import pandas as pd

    rng = np.random.default_rng(19)
    noise = rng.normal(size=len(y)) if noise is None else noise
    variants = {
        "original": y,
        "gain -12 dB": y * 0.25,
        "noise 20 dB": mix_noise(y, noise, 20),
        "noise 10 dB": mix_noise(y, noise, 10),
        "noise 0 dB": mix_noise(y, noise, 0),
    }
    rows = []
    for name, signal in variants.items():
        features = extract_features(signal, sr, offset, fmax)
        rows.append(dict(variant=name, clipping_fraction=float(np.mean(np.abs(signal) >= 0.999)), **features))
    return pd.DataFrame(rows), variants


def noise_robustness(events, family="resonance", noise=None, config=None, fmax=20000, snrs=(20, 10)):
    """Same held-out plates, clean vs noise-augmented training. No new independent observations."""
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, recall_score

    from .dataset import event_audio
    from .learning import evaluate, make_classifier, plate_table

    base, columns = plate_table(events, family)
    base = base[base.label.isin(["ok", "nok"])].reset_index(drop=True)
    _, _, folds = evaluate(events, family, bootstrap=0)
    rng = np.random.default_rng(812)
    variants = {"original": base}
    changed = {f"noise_{snr}dB": [] for snr in snrs}
    for row in events[events.plate_id.isin(base.plate_id)].to_dict("records"):
        y, sr = event_audio(row, config)
        background = rng.normal(size=len(y)) if noise is None else noise
        for snr in snrs:
            signal = mix_noise(y, background, snr)
            changed[f"noise_{snr}dB"].append(
                {**row, **extract_features(signal, sr, row["onset_offset_s"], fmax)}
            )
    for name, rows in changed.items():
        variants[name] = plate_table(pd.DataFrame(rows), family)[0]
        if variants[name].plate_id.tolist() != base.plate_id.tolist():
            raise RuntimeError("Niezgodna kolejność płytek w teście odporności.")
    y = (base.label == "nok").to_numpy(int)
    output = []
    for augmentation in (False, True):
        predictions = {name: np.full(len(base), np.nan) for name in variants}
        for fold in folds:
            tr = base.plate_id.isin(fold["train_plates"])
            te = base.plate_id.isin(fold["test_plates"])
            train_variants = list(variants.values()) if augmentation else [base]
            x = pd.concat([v.loc[tr, columns] for v in train_variants], ignore_index=True)
            labels = np.tile(y[tr], len(train_variants))
            # Preserve the total weight of each physical plate despite augmentation.
            weights = np.ones(len(labels)) / len(train_variants)
            model = make_classifier(family).fit(x, labels, logisticregression__sample_weight=weights)
            for name, table in variants.items():
                predictions[name][te] = model.predict_proba(table.loc[te, columns])[:, 1]
        for name, score in predictions.items():
            if not np.isfinite(score).all():
                raise RuntimeError("Brak części prognoz w teście odporności.")
            pred = score >= 0.5
            output.append(
                dict(
                    training="clean + noise" if augmentation else "clean only",
                    test_variant=name,
                    balanced_accuracy=balanced_accuracy_score(y, pred),
                    recall_nok=recall_score(y, pred, zero_division=0),
                    false_alarm_rate=float(np.mean(pred[y == 0])),
                    n_plates=len(base),
                    family=family,
                )
            )
    return pd.DataFrame(output)

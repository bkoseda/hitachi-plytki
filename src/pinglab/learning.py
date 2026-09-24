"""Exploratory models; the independent observation is a physical plate."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, recall_score
from .features import feature_columns


def plate_table(events, family='physics', aggregation='median'):
    if events.empty:
        raise ValueError('Brak uderzeń. Sprawdź segmentację.')
    if aggregation not in ('median','mean'):
        raise ValueError('Agregacja cech: median lub mean.')
    if (events.groupby('plate_id').label.nunique() > 1).any():
        raise ValueError('Sprzeczne etykiety tej samej płytki.')
    cols = feature_columns(events, family)
    if not np.isfinite(events[cols].to_numpy(float)).all():
        raise ValueError('Cechy zawierają NaN/Inf; popraw dane.')
    grouped = events.groupby('plate_id', sort=True)
    x = grouped[cols].agg(aggregation)
    x['label'] = grouped.label.first()
    x['n_hits'] = grouped.size()
    x['session_id'] = grouped.session_id.agg(lambda v: '|'.join(sorted(set(v))))
    return x.reset_index(), cols


def make_classifier(family='physics'):
    # Everything learned from the data stays INSIDE a fold.
    pre = Normalizer() if family == 'ast' else StandardScaler()
    return make_pipeline(pre, VarianceThreshold(1e-12),
                         LogisticRegression(C=.1, class_weight='balanced', max_iter=3000,
                                            solver='liblinear', random_state=42))


def _metrics(y, score, threshold=.5):
    pred = score >= threshold
    return dict(n_plates=int(len(y)), n_ok=int(np.sum(y==0)), n_nok=int(np.sum(y==1)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                recall_nok=float(recall_score(y,pred,zero_division=0)),
                false_alarm_rate=float(np.mean(pred[y==0])),
                roc_auc=float(roc_auc_score(y,score)),
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(), threshold=threshold)


def evaluate(events, family='physics', aggregation='median', by_session=False, seed=42, bootstrap=1000):
    table, cols = plate_table(events, family, aggregation)
    table = table[table.label.isin(['ok','nok'])].reset_index(drop=True)
    counts = table.label.value_counts()
    if counts.get('ok',0) < 3 or counts.get('nok',0) < 3:
        raise ValueError('Do oceny potrzeba co najmniej 3 niezależnych płytek OK i 3 NOK. '
                         'Dwie próbki służą tylko do porównania. Także 3+3 to wyłącznie wstępny test.')
    y = (table.label=='nok').astype(int).to_numpy()
    x = table[cols].to_numpy(float)
    if by_session:
        if table.session_id.str.contains('|',regex=False).any():
            raise ValueError('Płytki występują w wielu sesjach. Najpierw zaprojektuj rozłączny test sesji i płytek.')
        if table.session_id.nunique()<2:
            raise ValueError('Potrzebne co najmniej dwie sesje.')
        splits = list(LeaveOneGroupOut().split(x,y,groups=table.session_id))
    else:
        k = min(5,int(counts.min()))
        splits = list(StratifiedKFold(k,shuffle=True,random_state=seed).split(x,y))
    scores = np.full(len(y),np.nan)
    fold_id = np.full(len(y),-1)
    audit = []
    for fold,(train,test) in enumerate(splits):
        if len(np.unique(y[train])) != 2:
            raise ValueError('W zbiorze treningowym jednej z iteracji brakuje klasy. Zbierz obie klasy w sesjach.')
        model = make_classifier(family)
        model.fit(x[train],y[train])
        scores[test] = model.predict_proba(x[test])[:,1]
        fold_id[test] = fold
        audit.append(dict(fold=fold, train_plates=table.plate_id.iloc[train].tolist(),
                          test_plates=table.plate_id.iloc[test].tolist()))
    if not np.isfinite(scores).all():
        raise RuntimeError('Nie każda płytka otrzymała prognozę poza treningiem.')
    report = _metrics(y,scores)
    report.update(family=family,aggregation=aggregation,split='session' if by_session else 'plate',
                  warning='Wynik eksploracyjny. Próg 0.5 nie jest skalibrowanym prawdopodobieństwem wady. '
                          'Przed wdrożeniem potrzebny nowy zbiór, sesja i uzgodnione koszty błędów.')
    if bootstrap:
        rng = np.random.default_rng(seed)
        classes = [np.flatnonzero(y==k) for k in (0,1)]
        stats = []
        for _ in range(bootstrap):
            ids = np.concatenate([rng.choice(v,len(v),replace=True) for v in classes])
            stats.append(balanced_accuracy_score(y[ids],scores[ids]>=.5))
        report['balanced_accuracy_bootstrap_95'] = np.quantile(stats,[.025,.975]).tolist()
        report['interval_note'] = 'Bootstrap płytek na ustalonych prognozach OOF; nie obejmuje niepewności treningu ani zmiany fabryki.'
    predictions = table[['plate_id','label','n_hits','session_id']].copy()
    predictions['score_nok'] = scores
    predictions['prediction'] = np.where(scores>=.5,'nok','ok')
    predictions['fold'] = fold_id
    return report,predictions,audit


class NormalReference:
    """Distance to nearest known good plate. No defect label or calibrated probability."""
    def fit(self, x):
        x = np.asarray(x,float)
        if x.ndim != 2 or len(x)<3 or not np.isfinite(x).all():
            raise ValueError('Potrzeba >=3 poprawnych, skończonych wektorów różnych dobrych płytek.')
        self.scaler = StandardScaler().fit(x)
        self.reference = self.scaler.transform(x)
        d = cdist(self.reference,self.reference)/np.sqrt(x.shape[1])
        np.fill_diagonal(d,np.inf)
        self.reference_distances = d.min(axis=1)
        # Descriptive fence only, not a guaranteed false-positive rate.
        self.threshold = float(max(np.quantile(self.reference_distances,.95)*1.5,1e-6))
        return self

    def score(self,x):
        z = self.scaler.transform(np.asarray(x,float))
        return cdist(z,self.reference).min(axis=1)/np.sqrt(z.shape[1])


def anomaly_table(events, family='physics', reference_ids=None):
    table, cols = plate_table(events,family)
    if reference_ids is None:
        reference_ids = table.loc[table.label=='ok','plate_id'].tolist()
    reference_ids = set(reference_ids)
    if not reference_ids <= set(table.plate_id):
        raise ValueError('Nieznane ID wzorców.')
    ref = table.plate_id.isin(reference_ids)
    if (table.loc[ref,'label']!='ok').any():
        raise ValueError('Wzorce muszą mieć potwierdzoną etykietę ok.')
    model = NormalReference().fit(table.loc[ref,cols])
    scores = model.score(table[cols])
    # Avoid the trivial self-distance zero for reference plates.
    scores[ref] = model.reference_distances
    result = table[['plate_id','label','n_hits']].copy()
    result['reference'] = ref
    result['distance'] = scores
    result['exploratory_threshold'] = model.threshold
    result['unusual'] = scores > model.threshold
    return result, model


def permutation_check(events, family='physics', repeats=99, seed=43):
    """Shuffle whole plate labels, rerun the entire CV. Slow, optional sanity check."""
    actual = evaluate(events,family,bootstrap=0)[0]['balanced_accuracy']
    plates = events[['plate_id','label']].drop_duplicates()
    plates = plates[plates.label.isin(['ok','nok'])]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(repeats):
        mapping = dict(zip(plates.plate_id,rng.permutation(plates.label)))
        permuted = events.copy()
        permuted['label'] = permuted.plate_id.map(mapping).fillna('unknown')
        null.append(evaluate(permuted,family,bootstrap=0)[0]['balanced_accuracy'])
    return dict(actual=actual, null_scores=null,
                p_exploratory=(1+sum(s>=actual for s in null))/(1+repeats))

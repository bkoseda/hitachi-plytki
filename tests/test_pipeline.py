import json
import numpy as np
import pandas as pd
import pytest
from pinglab.demo import generate_demo
from pinglab.dataset import analyze_manifest,read_manifest
from pinglab.learning import evaluate,anomaly_table,plate_table
from pinglab.report import write_report


@pytest.fixture(scope='module')
def demo(tmp_path_factory):
    root=tmp_path_factory.mktemp('demo')
    manifest=generate_demo(root/'data',n_plates=12)
    f,q=analyze_manifest(manifest,root/'run',annotations=root/'data/annotations.json')
    return root,manifest,f,q


def test_end_to_end(demo):
    root,manifest,f,q=demo
    assert len(f)==36 and len(q)==13
    assert f.reviewed.all()
    metrics,pred,audit=evaluate(f,bootstrap=30)
    assert len(pred)==12
    for fold in audit:
        assert not set(fold['train_plates'])&set(fold['test_plates'])
    assert metrics['balanced_accuracy']>.75
    report=write_report(f,q,root/'report/report.html',(metrics,pred,audit),source_kind='synthetic')
    assert 'DANE SYNTETYCZNE' in report.read_text()
    assert (report.parent/'folds.json').exists()
    run=json.loads((root/'run/run.json').read_text())
    assert len(run['files'])==13


def test_two_plates_no_accuracy(demo):
    _,_,f,_=demo
    tiny=f[f.plate_id.isin(['SYN_001','SYN_012'])]
    with pytest.raises(ValueError,match='3 niezależnych'):
        evaluate(tiny)


def test_repeats_do_not_inflate_n(demo):
    _,_,f,_=demo
    dup=pd.concat([f]*4,ignore_index=True)
    a,cols=plate_table(f);b,_=plate_table(dup)
    np.testing.assert_allclose(a[cols],b[cols])
    assert evaluate(dup,bootstrap=0)[0]['n_plates']==12


def test_anomaly_reference_no_self_zero(demo):
    _,_,f,_=demo
    result,model=anomaly_table(f)
    assert (result.loc[result.reference,'distance']>0).all()
    assert np.isfinite(result.distance).all()
    with pytest.raises(ValueError):anomaly_table(f,reference_ids=['SYN_012'])


def test_session_split(demo):
    _,_,f,_=demo
    report,pred,audit=evaluate(f,by_session=True,bootstrap=0)
    assert report['split']=='session' and len(audit)==3
    for fold in audit:
        train=f[f.plate_id.isin(fold['train_plates'])]
        test=f[f.plate_id.isin(fold['test_plates'])]
        assert not set(train.session_id)&set(test.session_id)


def test_conflicting_labels_and_duplicates(demo,tmp_path):
    _,manifest,_,_=demo
    df=read_manifest(manifest)
    df['path']=df.resolved_path
    df.loc[1,'plate_id']=df.loc[0,'plate_id'];df.loc[1,'label']='nok'
    file=tmp_path/'bad.csv';df.to_csv(file,index=False)
    with pytest.raises(ValueError,match='sprzeczne'):read_manifest(file)
    df.loc[1,'plate_id']='different';df.loc[1,'path']=df.loc[0,'path']
    df.to_csv(file,index=False)
    with pytest.raises(ValueError,match='dwukrotnie'):read_manifest(file)


def test_low_sample_rate_rejects_incomparable_band(tmp_path):
    import soundfile as sf
    from pinglab.demo import resonator
    sf.write(tmp_path/'test.wav',resonator(sr=16000),16000)
    pd.DataFrame([dict(path='test.wav',plate_id='P1',label='ok',session_id='S1')]).to_csv(tmp_path/'manifest.csv',index=False)
    with pytest.raises(ValueError,match='ponad pasmem'):
        analyze_manifest(tmp_path/'manifest.csv',tmp_path/'out')
    f,q=analyze_manifest(tmp_path/'manifest.csv',tmp_path/'out2',fmax=7500)
    assert len(f)>0

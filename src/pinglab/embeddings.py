"""Optional frozen AudioSet AST. CPU by default, no upload and no fine-tuning."""
from pathlib import Path
import hashlib
import json
import numpy as np
import librosa
from .dataset import event_audio

MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'


class ASTEncoder:
    def __init__(self, cache_dir='models/ast', offline=False, device='cpu'):
        import torch
        from transformers import ASTFeatureExtractor, ASTModel
        torch.set_num_threads(min(8,torch.get_num_threads()))
        self.device = device
        self.cache_dir = str(Path(cache_dir).resolve())
        args = dict(cache_dir=self.cache_dir,local_files_only=offline)
        self.processor = ASTFeatureExtractor.from_pretrained(MODEL_ID,**args)
        self.model = ASTModel.from_pretrained(MODEL_ID,**args).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.revision = getattr(self.model.config,'_commit_hash',None) or 'unknown'

    def encode(self,y,sr):
        import torch
        y = np.asarray(y,dtype=np.float32)
        if y.ndim!=1 or not np.isfinite(y).all() or len(y)/sr<.05:
            raise ValueError('AST oczekuje mono, >=50 ms, bez NaN/Inf.')
        if len(y)/sr>10:
            raise ValueError('AST: podaj wycięte uderzenie <=10 s, nie całe nagranie.')
        if np.max(np.abs(y))<1e-7:
            raise ValueError('AST: cisza nie jest uderzeniem.')
        y = y - y.mean()
        y = y / max(float(np.max(np.abs(y))),1e-8) * .95
        if sr!=16000:
            y=librosa.resample(y,orig_sr=sr,target_sr=16000)
        inputs=self.processor(y,sampling_rate=16000,return_tensors='pt')
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        with torch.inference_mode():
            vector=self.model(**inputs).pooler_output[0].cpu().numpy()
        vector=vector/max(float(np.linalg.norm(vector)),1e-12)
        if vector.shape!=(768,) or not np.isfinite(vector).all():
            raise RuntimeError('Nieprawidłowy embedding AST.')
        return vector.astype(np.float32)

    def add_to_events(self,events,cache='data/cache/ast',config=None):
        cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
        vectors=[]
        for row in events.to_dict('records'):
            # Hash actual decoded input too: changed channel/config cannot reuse an old vector.
            y,sr=event_audio(row,config)
            key=hashlib.sha256(y.tobytes()+str(sr).encode()+self.revision.encode()+b'peak095-v1').hexdigest()
            file=cache/f'{key}.npy'
            if file.exists():
                vector=np.load(file,allow_pickle=False)
            else:
                vector=self.encode(y,sr)
                np.save(file,vector,allow_pickle=False)
            if vector.shape!=(768,) or not np.isfinite(vector).all():
                raise ValueError(f'Uszkodzony cache: {file}')
            vectors.append(vector)
        if not vectors:
            raise ValueError('Brak uderzeń do AST.')
        result=events.copy()
        import pandas as pd
        result=pd.concat([result.reset_index(drop=True),
                          pd.DataFrame(np.stack(vectors),columns=[f'ast_{i:03d}' for i in range(768)])],axis=1)
        (cache/'model.json').write_text(json.dumps(dict(model=MODEL_ID,revision=self.revision,
                      sampling_rate=16000,dimensions=768,normalization='peak 0.95 + model defaults + L2'),indent=2))
        return result

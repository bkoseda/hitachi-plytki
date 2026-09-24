# Dlaczego te metody?

Sprawdzone źródła: 24.09.2026. Wnioski poniżej są decyzjami dla tego warsztatu, nie obietnicą jakości u klienta.

## Najbliższe badanie

[Acoustic Structural Integrity Assessment of Ceramics using Supervised Machine Learning and Uncertainty-Based Rejection](https://kdd.org/exploration_files/p105-AI4Manufacturing_paper3.pdf), SIGKDD Explorations, 2022.
Autorzy zebrali 31 ceramicznych talerzy (16 pękniętych, 15 całych) i po kilka uderzeń na sztukę.
To znacznie bliższy przypadek niż rozpoznawanie mowy. Publikacja opisuje cechy akustyczne i małe klasyfikatory;
jednocześnie wskazuje ograniczenia małej próby i przejścia z laboratorium do przemysłu.
Nie znalazłem w publikacji publicznego archiwum WAV. Nie odtwarzamy jej zgłoszonej skuteczności i nie
przenosimy jej wyniku na płytki klienta. Nasz podział zawsze oddziela fizyczne płytki.

## Dane do ćwiczeń

| Źródło | Co daje | Jak używamy |
|---|---|---|
| [RealImpact — autorzy i demo](https://samuelpclarke.com/realimpact/) | Rzeczywiste odpowiedzi na uderzenia, różne pozycje mikrofonów | Pobrano 5 WAV publicznego demo, 48 kHz, po 1.5 s; sprawdzono dekoder, segmentację i raport |
| [Repo RealImpact](https://github.com/samuel-clarke/RealImpact) | Kod i odnośniki do większych archiwów obiektów, w tym ceramicznych | Skrypt pobiera tylko małe demo; nie ściąga całego wielogigabajtowego zbioru |
| [Coin-tap / torch-tapnet](https://github.com/PPhub-hy/torch-tapnet) | Pokrewne badanie detekcji defektów po stuknięciu | Link do danych prowadzi do Baidu z kodem dostępu; nie włączono danych do działającego demo |
| `pinglab.demo` | 30 syntetycznych rezonatorów, 3 uderzenia na „płytkę”, 20 OK / 10 NOK | Deterministyczny test kodu i instrukcji, nie benchmark ceramiki |

**RealImpact nie ma tu etykiet OK/NOK.** Publiczny przykład przedstawia pole dźwiękowe obiektu z różnych pozycji;
wszystkie pięć plików mają wspólny `plate_id`. Nie wolno z tych pięciu mikrofonów robić pięciu niezależnych detali.
Nagrania są odpowiedziami po dekonwolucji, więc różnią się od surowego mikrofonu w hali.
Pochodzenie i SHA-256: `data/external/realimpact/provenance.json`. Kopie pozostają lokalne, poza Git.
Repo źródłowe zawiera licencję MIT dla oprogramowania; nie traktujemy jej automatycznie jako odrębnej zgody
na publiczną redystrybucję nagrań demonstracyjnych. Przed zewnętrznym udostępnieniem sprawdź warunki danych.

Szukałem także materiałów na Kaggle, Zenodo i Mendeley. Nie znalazłem małego, bezpośrednio dostępnego i
jednoznacznie opisanego zbioru nagrań ceramicznych płytek OK/NOK, który można uczciwie uznać za walidację tego projektu.
Dźwięk rozbijania talerza nie jest odpowiednim zamiennikiem testu uderzeniowego już istniejącego pęknięcia.

## Kolejność metod

1. **Rezonans + wybrzmiewanie.** Niewiele cech, można wyjaśnić różnicę na wykresie. Częstotliwości,
   udział energii w pasmach i zanik zależą również od geometrii, podparcia i pozycji uderzenia.
2. **MFCC / mel.** Krótki opis barwy albo rozkładu energii. Korzystamy z
   [librosa 0.11](https://librosa.org/doc/0.11.0/feature.html). MFCC pomija współczynnik c0; amplituda jest
   normalizowana wyłącznie w kopii służącej do ekstrakcji. Kontrola jakości zachowuje oryginalny poziom.
3. **Regresja logistyczna.** Mały model z silną regularyzacją i równoważeniem klas. Jedna obserwacja = jedna płytka.
   Preprocessing znajduje się wewnątrz każdego podziału, zgodnie z zasadami
   [scikit-learn: unikanie przecieku](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
4. **Odległość od dobrych wzorców.** Bez uczenia granicy OK/NOK. Pomaga wybrać nagrania do obejrzenia;
   nie rozróżnia wady od zmiany procesu pomiarowego. Próg jest wyłącznie eksploracyjny.
5. **Zamrożony AST.** Drugi opis dźwięku do porównania z prostszą bazą; nie dotrenowujemy sieci.

## Gotowy model i plan awaryjny

[AST — dokumentacja modelu](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)
i [wagi MIT / AudioSet](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593).
Warsztat używa 768-wymiarowego `pooler_output`, normalizacji L2 i CPU. Wagi safetensors zajmują około 346 MB.
Wejście jest resamplowane do 16 kHz; pasmo ponad 8 kHz przepada. Krótki hit jest dopełniany ekstraktorem
zgodnie z ustawieniami modelu. To potencjalne niedopasowanie do AudioSet, które trzeba sprawdzić empirycznie.
Model, hash rewizji i parametry są zapisane w cache. Pierwsze pobranie wymaga sieci; późniejsze wywołania
mogą mieć `offline=True`. Żadne audio klienta nie jest wysyłane.

Inne rozsądne kandydatury do przyszłego porównania: [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)
i [LAION CLAP](https://github.com/LAION-AI/CLAP). Nie są zainstalowanymi backendami tego repo.
Ogólny model audio może rozumieć barwę i zdarzenia, lecz nie musi zachowywać subtelnych rezonansów wady.
Opisy tekstowe „cracked ceramic” nie zastępują etykiet i testów.
Przy kilkudziesięciu niezależnych detalach pierwszym krokiem nadal jest zamrożona reprezentacja + mały model.
Odmrożenie końcowych warstw rozważyć dopiero po zebraniu większej, rozdzielonej po detalach i sesjach próby.

## Czego celowo nie robimy

Nie uczymy dużej sieci od zera. Nie powielamy uderzeń jako nowych płytek. Nie wybieramy progu na tym samym
zbiorze, na którym raportujemy skuteczność. Nie interpretujemy PCA jako testu modelu. Nie stosujemy domyślnego
agresywnego odszumiania. Nie dobieramy najlepszego modelu na syntetycznych danych dla rzeczywistego klienta.

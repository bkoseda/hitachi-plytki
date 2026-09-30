# Nowe paczki klienta · 30.09.2026

Nowe dane są wyraźnie inne. Folder **Mic Enhancements Off** wygląda lepiej pod kątem zachowania tła: 0/3 plików ma ciąg dokładnych zer dłuższy niż 100 ms. W **LAB TEST** takie odcinki występują w 13/13 plików, trwają do około 2,4 s; mediana udziału zer w pliku to około 56%. Sam sygnał nie rozstrzyga, czy to bramka, odszumianie, edycja czy przerwy toru.

Uderzenia są teraz rzadsze: typowy odstęp wynosi około 2,1–2,2 s wobec około 0,28 s wcześniej. To daje miejsce na ogon dźwięku, ale cyfrowe wyciszanie nadal ogranicza badanie zaniku. Nie mamy jeszcze dowodu, że dłuższy ogon poprawi klasyfikację.

**Najważniejszy wynik:** model MFCC + regresja uczony tylko na wcześniejszych danych trafia 12/13 plików LAB, lecz odrzuca oba nowe OK z Enhancements Off. Potrzebujemy NOK nagranego w tych samych warunkach z wyłączonymi ulepszeniami. Nie należy cofać tej zmiany po to, żeby poprawić wynik starego modelu.

## Jak testujemy

Analizujemy wszystkie 61 plików w kontroli jakości. Model pęknięć obejmuje 42 nagrania z roboczymi etykietami OK/NOK; tło, OLD o nieznanej tożsamości i Bitten są poza głównym treningiem. To nie oznacza, że wyszczerbienie jest niewykrywalne — jest innym rodzajem defektu.

Detektor szuka nagłych wzrostów krótkookresowego RMS: ramka 10 ms, krok 5 ms, próg zależny od tła, minimalny odstęp 120 ms. Odrzuca zbyt słabe i niepełne okna. W nowych danych przyjął 258 uderzeń; nie traktujemy ich jako ręcznie potwierdzonych trafień. Model dostaje do 10 równomiernie wybranych uderzeń z nagrania: każde to 20 ms przed + 150 ms po początku, z normalizacją amplitudy. Decyzja o nagraniu to mediana wyników uderzeń, próg 0,5.

Porównujemy physics (15 cech), MFCC (26), ich połączenie (41), zamrożony AST (768). LR: C=0,1. XGBoost: 60 drzew, głębokość 2, bez strojenia na nowych danych. Skalowanie i model uczą się tylko na części treningowej. Kontrola QC używa jedynie poziomu, udziału ciszy i długości pliku, bez widma i kodeka.

W teście łączonym odkładamy całe grupy uznane w danym scenariuszu za płytki: jedną NOK i jedną OK. Dla powtarzających się odłożeń uśredniamy wynik nagrania. Są to wyniki eksploracyjne, zależne od niepotwierdzonych ID; wiele nagrań tej samej sztuki nie jest wieloma niezależnymi płytkami. Scenariusze nie wyczerpują wszystkich możliwych przypisań. Nowe OK roboczo łączymy z OK_1, ale klient może wskazać inne ID.

## Do potwierdzenia

1. Czy nowe `BAD/BAD2` to ta sama fizyczna płytka co wcześniejsze `Bad_2`, czy nowa pęknięta sztuka? Prosimy o krótkie przypisanie: folder → ID płytki → pęknięcie / wyszczerbienie / OK.
2. Czy pliki w `LAB TEST` powstały przed czy po wyłączeniu ulepszeń? Czy `LAB` oznacza inne pomieszczenie, aplikację lub mikrofon? W każdym z tych 13 plików nadal są długie odcinki dokładnego zera.
3. Czy `OK` z LAB i `OK` z Mic Enhancements Off to ta sama płytka? Która z poprzednich OK? Czy końcówki 2 i 3 oznaczają powtórzenie nagrania, a nie nową sztukę?
4. Co oznacza `angle` (kąt mikrofonu czy uderzenia) oraz `single point / one point` (miejsce uderzania czy podparcie)? W tych ZIP-ach nie znaleźliśmy nazw `no freq` ani `no frequency`.

Pełne tabele i odsłuch: `reports/runs/client-update-20260930/START.html`. Notebook: `notebooks/client/05_nowe_paczki_20260930.ipynb`. Wyniki CSV w tym samym katalogu co HTML. Oryginały w `assets/3-batch-enhancements-off` i `assets/4-batch-lab-test`; wszystkie cztery dostawy są pod wspólnym `assets`.

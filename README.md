# Platforma analityki biomechanicznej i telemetrii end-to-end


Kompleksowy projekt analityki danych ruchowych (Time-Series & Sensor Fusion), integrujący dane telemetryczne IoT (binarne pliki Garmin .FIT / .ZIP) z niestrukturyzowanymi strumieniami wideo (Computer Vision). System realizuje pełny potok analityczny: od ekstrakcji i czyszczenia danych (ETL), przez wyznaczanie wskaźników kinematycznych (KPI), aż po automatyczne reguły decyzyjne i raport w aplikacji webowej.


<br>


## Prezentacja działania

> **Materiały źródłowe:** Nagrania wideo wykorzystane do testów i demonstracji algorytmów pochodzą z kanału **BioMechanic** w serwisie YouTube.

### 1. Widok z boku (Płaszczyzna strzałkowa)
Ocena lądowania, pochylenia tułowia, oscylacji pionowej i sprężystości kończyny (Leg Stiffness).

<p align="center">
  <video src="static/video_record_side.mp4" controls width="750"></video>
</p>

### 2. Widok od tyłu (Płaszczyzna czołowa)
Śledzenie stabilności miednicy, pronacji/supinacji stóp oraz asymetrii obciążenia lewa/prawa noga.

<p align="center">
  <video src="static/video_record_rear.mp4" controls width="750"></video>
</p>



> **Wytyczne dotyczące nagrań wejściowych:**  
> Aby zapewnić poprawną estymację punktów anatomicznych, nagranie musi obejmować **całą sylwetkę biegacza** i być zarejestrowane stabilnie w osi ruchu: w **widoku z boku** z prawego profilu (widoczne całe ciało od głowy po kontakt stóp z podłożem), a w **widoku od tyłu** centralnie na wysokości miednicy/kolan (widoczne punkty od barków po zapiętki i podeszwy butów).

<br>


## 1. Problem Biznesowy i Kontekst Analityczny


W analizie biomechaniki sportowej proces wnioskowania cierpi na brak integracji odizolowanych źródeł danych:

* **Silosy danych (Data Silos):** Urządzenia telemetryczne (np. Garmin) rejestrują czas kontaktu z podłożem (GCT) i kadencję, ale działają w izolacji od przestrzennego układu anatomicznego biegacza.
* **Niewidoczne anomalie dynamiczne:** Kluczowe wady biomechaniczne (np. overstriding, dynamiczny opad miednicy, kompensacyjna pronacja) trwają ułamki sekund i umykają tradycyjnym agregacjom statystycznym bez powiązania z obrazem.
* **Wysoki koszt aparatury pomiarowej:** Dostęp do laboratoryjnych systemów motion-capture (np. Vicon) jest zaporowy cenowo dla klubów i biegaczy amatorów.


<br>


### Rozwiązanie Analityczne:
Aplikacja wdraża zautomatyzowany potok analityczny, który:
1. Integruje szeregi czasowe z wearables z przestrzenną detekcją punktów kluczowych (**MediaPipe Pose**).
2. Normalizuje i czyści surowe sygnały za pomocą buforów kołowych i filtracji odchyleń skrajnych.
3. Oblicza zestaw wskaźników biomechanicznych (KPI) w czasie rzeczywistym bezpośrednio na nagraniu (**HUD Overlay**).
4. Generuje zautomatyzowany raport decyzyjny (**Prescriptive Analytics**) wskazujący przyczyny źródłowe asymetrii i ryzyka kontuzji.


<br>


## 2. Architektura Przepływu Danych i Pipeline ETL
```
Dane Wejściowe (Plik wideo + Garmin .FIT + Wzrost biegacza)
↓
Data Ingestion (Dekodowanie binarnego pliku .FIT przez fitparse & kalibracja skali px→cm)
↓
Feature Extraction (Ekstrakcja współrzędnych stawów X, Y przez MediaPipe Pose)
↓
Silnik Analityczny KPI (Obliczanie kątów kolana/tułowia, oscylacji, pronacji stóp i asymetrii)
↓
Data Cleaning & Smoothing (Redukcja szumów buforami kołowymi FIFO & filtracja skoków)
↓
Data Delivery (Nakładanie wizualnych metryk HUD na klatki wideo za pomocą OpenCV)
↓
Decision Intelligence (Reguły biznesowe, wykrywanie anomalii i klasyfikacja ryzyka kontuzji)
↓
Raportowanie BI (Prezentacja syntetycznych wskaźników i rekomendacji w panelu Flask)
```


<br>


## 3. Przetwarzanie Danych, Inżynieria Cech (Feature Engineering) i Metodyka Obliczeniowa

Warstwa analityczna przekształca surowe szeregi czasowe współrzędnych (X, Y) w ustrukturyzowany zbiór wskaźników kinematycznych odpornych na szum pomiarowy.


### 1) Normalizacja Przestrzenna i Czyszczenie Szeregów Czasowych
**Dynamiczne skalowanie metryczne (Feature Scaling):** Aby uniezależnić analizę od odległości biegacza od obiektywu, wprowadzono antropometryczną normalizację jednostek (piksele -> centymetry). Wzorcem kalibracyjnym jest odcinek tułowia (bark–biodro), stanowiący biologiczny niezmiennik (~30% wzrostu użytkownika):

$$\text{px to cm} = \frac{\text{USER HEIGHT CM} \times 0.3}{\Vert{}\mathbf{p}_{\text{bark}} - \mathbf{p}_{\text{biodro}}\Vert{}_2}$$

**Wygładzanie sygnału i redukcja drżenia (Noise Reduction):** Współrzędne węzłów szkieletowych oraz obliczane kąty poddawane są wygładzaniu średnią ruchomą z wykorzystaniem buforów kołowych FIFO (`collections.deque`, okno N=5). Zapobiega to fałszywym alertom przy detekcji faz kroku.


<br>


### 2) Płaszczyzna Strzałkowa (Rzut Boczny) – Kinematyka i Fuzja Sensorów
* **Obliczanie kątów wewnętrznych stawów (Vector Geometry):** Wyznaczane z iloczynu skalarnego wektorów anatomicznych z zabezpieczeniem numerycznym (`clip` do [-1.0, 1.0] oraz epsilon = 1e-6):
  $$\theta = \arccos\left(\text{clip}\left(\frac{\mathbf{ba} \cdot \mathbf{bc}}{\Vert\mathbf{ba}\Vert \Vert\mathbf{bc}\Vert + 10^{-6}}, -1.0, 1.0\right)\right)$$

  
* **Detekcja faz kroku (Event-Based Time-Series Slicing):** Moment lądowania (*initial contact*) izolowany jest poprzez wykrycie wyhamowania ruchu stopy w przód na bazie historii współrzędnych X stawu skokowego.

  
* **Detekcja overstridingu (Braking Force KPI):** Wartość kąta wyprostu kolana w momencie kontaktu > 170° flagowana jest jako anomalia techniczna (lądowanie przed środkiem ciężkości, generujące szkodliwe siły hamujące).

  
* **Fuzja sensoryczna (Leg Stiffness Index):** Wskaźnik łączący telemetrię zegarka z analizą wideo. Zestawia czas kontaktu z podłożem (GCT z pliku FIT w milisekundach) z dynamicznym zakresem ugięcia kolana (180° - kąt_minimalny z Computer Vision):
  $$\text{Stiffness} = \frac{50000}{GCT \times (180 - \theta_{\min})}$$

  
* **Oscylacja pionowa (Robust Dispersion Metric):** Wyznaczana z rozstępu międzycentylowego (P95 - P5) trajektorii pionowej biodra przeliczonego na centymetry, co eliminuje pojedyncze szumy detekcji.


<br>


### 3) Płaszczyzna Czołowa (Rzut od Tyłu) – Segmentacja Osi i Analiza Asymetrii
* **Segmentacja fazy podparcia (Stance Phase Filtering):** Pronacja i supinacja analizowane są wyłącznie w fazie obciążenia stopy, wyodrębnianej adaptacyjnym progowaniem percentylowym trajektorii Y pięty i stawu skokowego (eliminacja fazy lotu).

  
* **Kąt nachylenia stopy (Robust Directional Angularity):** Odchylenie wektora pięta–staw skokowy od pionu modelowane za pomocą `atan2`:
  $$\text{Tilt} = \text{degrees}(\text{arctan2}(dy, dx)) + 90^\circ$$
  * Odchylenie < -12°: Nadmierna pronacja.
  * Odchylenie > +12°: Supinacja kompensacyjna.
 
    
* **Opadanie miednicy (Pelvic Drop):** Kąt nachylenia wektora łączącego lewe i prawe biodro względem osi poziomej.

  
* **Dwuścieżkowa klasyfikacja asymetrii (Root-Cause Analysis):**
  * **Asymetria strukturalna (Geometryczna):** Różnica w szczytowej wysokości uniesienia pięt w fazie lotu (> 3%) wskazuje na ograniczenia ruchomości lub dysproporcję siłową.
  * **Asymetria czasowo-kinetyczna (Telemetryczna):** Symetryczny tor ruchu przy nierównym czasie kontaktu z podłożem (|50 - Balans GCT| > 2%) wskazuje na odruchowe odciążanie jednej z kończyn.


<br>


## 4. Benchmark i Diagnostyka Biomechaniczna

Każda metryka posiada zdefiniowane progi tolerancji. Ich przekroczenie generuje automatyczne alerty w raporcie końcowym:

| Wskaźnik (KPI) | Płaszczyzna | Źródło Danych | Przedział Prawidłowy | Próg Alarmowy | Wpływ na Efektywność / Ryzyko |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kadencja (SPM)** | Boczna | AI / Garmin FIT | 170 – 185 SPM | < 165 lub > 195 SPM | Nieoptymalna długość kroku, straty energii |
| **Kąt kolana przy lądowaniu** | Boczna | MediaPipe | 150° – 168° | > 170° (Overstriding) | Siły hamujące, przeciążenie stawu rzepkowo-udowego |
| **Oscylacja pionowa** | Boczna | MediaPipe (cm) | 6.0 – 9.5 cm | > 10.0 cm | Marnowanie energii na wektor pionowy zamiast poziomy |
| **Pochylenie tułowia** | Boczna | MediaPipe | 5° – 10° | < 3° lub > 12° | Niewłaściwe wykorzystanie grawitacji, przeciążenia lędźwiowe |
| **Leg Stiffness Index** | Boczna | Sensor Fusion | 3.5 – 6.0 | < 3.5 (Low Stiffness) | Zapadanie się w podporze, wydłużony kontakt z podłożem |
| **Kąt ugięcia ramion** | Boczna | MediaPipe | 80° – 95° | < 60° lub > 105° | Napięcia obręczy barkowej, zaburzenie rytmu wahadła |
| **Pronacja / Supinacja** | Tylna | MediaPipe | -12° do +12° | < -12° lub > +12° | Ryzyko kontuzji rozcięgna podeszwowego / ścięgna Achillesa |
| **Przechylenie miednicy** | Tylna | MediaPipe | < 4.0° | > 5.0° (Pelvic Drop) | Niewydolność mięśnia pośladkowego średniego |
| **Balans GCT (L/R)** | Tylna | Garmin FIT | 49.0% – 51.0% | Odchylenie > 2.0% | Nierównomierne przenoszenie obciążeń uderzeniowych |


<br>


## 5. Decision Intelligence & Actionable Insights

Zamiast surowych wykresów, system implementuje warstwę analityki preskryptywnej. Zidentyfikowane odchylenia są natychmiast przekładane na gotowe wskazówki trenerskie:

```python
# Przykład reguły w silniku analityki preskryptywnej (side_view_analysis.py)
if avg_knee_angle > 170:
    advice.append({
        "issue": "Overstriding (lądowanie z nadmiernie wyprostowanym kolanem)",
        "consequence": "Lądowanie przed środkiem ciężkości generuje siły hamujące i przeciąża staw kolanowy.",
        "fix": "Skróć krok i zwiększ kadencję o około 5%."
    })
```


<br>


## 6. Stos Technologiczny
* **Język bazowy**: Python 3.10+
* **Analiza i Modelowanie Danych**: Pandas, NumPy, SciPy
* **ETL & Data Ingestion**: fitparse (dekodowanie binarnego formatu Garmin FIT), OpenCV
* **Feature Extraction & Pose Estimation**: MediaPipe Pose (Model Complexity 2)
* **Prezentacja & Dashboard**: Flask, Jinja2, HTML5/CSS, HUD Overlay


<br>


## 7. Architektura Projektu
```
fitform-gait-analytics/
├── static/                   # Wyjściowe nagrania z naniesionym HUD i assety
│   ├── .gitkeep
│   ├── result_running_rear.mp4
│   └── result_running_side.mp4
├── templates/
│   ├── index.html            # Formularz uploadu wideo, danych FIT i wzrostu
│   └── results.html          # Panel podsumowujący biomechanikę i zalecenia
├── uploads/                  # Bufor przesłanych plików wejściowych
├── sample_data/              # Przykładowe pliki wideo i telemetrii
├── app.py                    # Główna aplikacja Flask i routing zapytań
├── rear_view_analysis.py     # Pipeline analizy rzutu od tyłu (stopy, miednica)
├── side_view_analysis.py     # Pipeline analizy rzutu bocznego
├── requirements.txt          # Zależności biblioteczne
└── README.md                 # Dokumentacja techniczna projektu
```


<br>


## 8. Jak uruchomić projekt

Klonowanie repozytorium:

```bash
git clone [https://github.com/laura-szczerbowska/GaitAnalytics.git](https://github.com/laura-szczerbowska/GaitAnalytics.git)
cd GaitAnalytics
```
Instalacja zależności:
```bash
pip install -r requirements.txt
```
Uruchomienie serwera lokalnego
```bash
python app.py
```
Aplikacja będzie dostępna pod adresem: http://127.0.0.1:5000


<br>


## 9. Rozwój Projektu
* Automatyczna kalibracja przesunięcia czasowego pomiędzy wideo a plikiem .FIT na bazie wykrywania pierwszego kroku.
* Wdrożenie akceleracji GPU (CUDA / TensorRT) dla ekstrakcji klatek MediaPipe w rozdzielczości 4K przy 60/120 FPS.
* Integracja rzutu czołowego (Front View) do oceny dynamicznej koślawości kolan (Knee Valgus).
* Eksport raportu sesji biegowej do formatu PDF z wykresami zmian kątów w czasie.


<br>


## 10. Kluczowe Wnioski Inżynieryjne
* **Jakość danych a modelowanie (Garbage In, Garbage Out)**: Surowe współrzędne bez progowania generują szum przy dynamicznym ruchu kończyn. Zastosowanie buforów FIFO i progowania odległościowego ustabilizowało wariancję odczytów kątów o ponad 30%.
* **Wartość analityczna z Sensor Fusion**: Samo wideo przy standardowym klatkażu nie pozwala na precyzyjny pomiar milisekundowego kontaktu z podłożem, a zegarek Garmin nie widzi kątów w stawach. Fuzja obu strumieni umożliwiła wyznaczenie wskaźnika sztywności (Leg Stiffness Index).
* **Automatyzacja wnioskowania**: Zastąpienie surowych tabel zautomatyzowanymi regułami diagnostycznymi pozwala na natychmiastową interpretację wyników bezpośrednio po przetworzeniu sesji.

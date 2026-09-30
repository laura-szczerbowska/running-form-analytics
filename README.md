<div align="right">
  <strong>English</strong> | <a href="./README.pl.md">Polski</a>
</div>

# End-to-End Biomechanical Gait Analytics & Telemetry Platform

A comprehensive motion data analytics project integrating IoT telemetry data (binary Garmin .FIT / .ZIP files) with unstructured video streams (Computer Vision). The system implements a full analytics pipeline: from data extraction and cleaning (ETL), through kinematic key performance indicator (KPI) estimation, to automated prescriptive decision rules and interactive dashboard reporting.


<br>


## Demo Presentation

> **Source Materials:** Video recordings used for algorithm demonstration were sourced from YouTube channels. Testing and validation were conducted on private treadmill recordings.


<br>


### 1) Sagittal Plane (Side View)
Evaluation of footstrike angle, torso lean, vertical oscillation, and limb elasticity (Leg Stiffness).

https://github.com/user-attachments/assets/a98aabef-e1fd-45e4-8d80-2c9be2a3783c


<br>


### 2) Frontal Plane (Rear View)
Tracking pelvic stability (drop), foot pronation/supination, and left/right ground contact asymmetry.

https://github.com/user-attachments/assets/26275ffc-46dd-4f2d-8830-8d3b637db2e8

> **Input Video Guidelines:**  
> To ensure accurate keypoint tracking, the footage must capture the **runner's entire body** within a stable field of view: the **side view** must be recorded directly from the right profile (head to full foot-ground contact), while the **rear view** must be centered behind the runner at hip/knee level (shoulders down to shoe heels and soles).


<br>


## 1. Business Problem & Analytical Context

In sports biomechanics, actionable inference frequently suffers from disconnected data silos:

- **Data Silos:** Telemetry devices (e.g., Garmin) accurately record ground contact time (GCT) and cadence but lack spatial and anatomical context.
- **Hidden Dynamic Anomalies:** High-risk kinematic defects (e.g., overstriding, dynamic pelvic drop, compensatory overpronation) occur within fractions of a second and escape traditional aggregated summary metrics.
- **Prohibitive Hardware Costs:** Traditional motion-capture laboratories (e.g., Vicon) remain cost-prohibitive for amateur runners, coaches, and local clubs.


<br>


### Analytical Solution:
This application implements an automated end-to-end analytics pipeline that:
1. Merges wearable time-series telemetry with computer vision keypoint detection (**MediaPipe Pose**).
2. Cleans and normalizes noisy spatial signals using circular FIFO buffers and statistical outlier filtering.
3. Computes domain-specific kinematic KPIs in real time directly on video frames (**HUD Overlay**).
4. Delivers automated root-cause diagnostics (**Prescriptive Analytics**) identifying injury risks and movement inefficiencies.


<br>


## 2. Architecture & ETL Pipeline Flow
```
Input Data (Video File + Garmin .FIT + Runner Height)
↓
Data Ingestion (Binary .FIT decoding via fitparse & metric calibration px→cm)
↓
Feature Extraction (X, Y anatomical joint coordinates via MediaPipe Pose)
↓
KPI Analytics Engine (Joint angles, vertical oscillation, pronation & asymmetry)
↓
Data Cleaning & Smoothing (FIFO circular buffers & delta spike filtering)
↓
Data Delivery (Real-time HUD metric overlay rendering via OpenCV)
↓
Decision Intelligence (Rule-based heuristics, anomaly detection & risk scoring)
↓
BI Reporting (Interactive dashboard synthesis and recommendations in Flask)
```

<br>


## 3. Data Processing, Feature Engineering & Computational Methods

The analytical layer transforms raw coordinate time-series $(X, Y)$ into robust kinematic indicators resilient to sensor noise.


<br>


### 1) Spatial Normalization & Signal Conditioning

- **Dynamic Metric Scaling (Feature Scaling):** To eliminate perspective distortion and varying camera distances, metric calibration converts pixel measurements to centimeters using the torso segment (shoulder–hip) as an anthropometric invariant (~30% of total height):


$$\text{px to cm} = \frac{\text{USER HEIGHT CM} \times 0.3}{\Vert\mathbf{p}_{\text{shoulder}} - \mathbf{p}_{\text{hip}}\Vert_2}$$


- **Signal Smoothing & Jitter Reduction (Noise Reduction):** Skeletal landmark coordinates and derived joint angles are smoothed using a moving average computed over FIFO circular buffers (`collections.deque`, window $N=5$), preventing false-positive footstrike detections.


<br>


### 2) Sagittal Plane (Side View) — Kinematics & Sensor Fusion

- **Joint Angle Computation (Vector Geometry):** Derived from vector dot products with numerical stability clipping (bounded to $[-1.0, 1.0]$ with an epsilon of $10^{-6}$):


$$\theta = \arccos\left(\text{clip}\left(\frac{\mathbf{ba} \cdot \mathbf{bc}}{\Vert\mathbf{ba}\Vert \Vert\mathbf{bc}\Vert + 10^{-6}}, -1.0, 1.0\right)\right)$$


- **Gait Event Detection (Time-Series Slicing):** Initial ground contact is isolated by identifying horizontal deceleration points in the ankle's X-coordinate trajectory.

- **Overstriding Anomaly Detection (Braking Force KPI):** Knee extension angles exceeding 170° at initial ground strike are flagged as overstriding, indicating high braking forces.

- **Sensor Fusion (Leg Stiffness Index):** Combines wearable telemetry with computer vision by indexing millisecond Ground Contact Time (GCT from Garmin) against dynamic knee deflection ($180^\circ - \theta_{\min}$ from OpenCV):


$$\text{Stiffness} = \frac{50000}{\text{GCT} \times (180 - \theta_{\min})}$$


- **Vertical Oscillation (Robust Dispersion Metric):** Quantified using the interpercentile range (P95 - P5) of vertical hip displacement converted to centimeters to mitigate tracking jitter.


<br>


### 3) Frontal Plane (Rear View) — Segment Alignment & Asymmetry

- **Stance Phase Filtering:** Pronation and supination are isolated exclusively during weight-bearing stance, identified via dynamic thresholding of vertical heel and ankle trajectories.

- **Foot Tilt Angle (Directional Angularity):** Angular deviation of the heel–ankle vector relative to the vertical axis modeled using `atan2`:


$$\text{Tilt} = \text{degrees}(\text{arctan2}(dy, dx)) + 90^\circ$$


  * Tilt < -12°: Excessive overpronation.
  * Tilt > +12°: Compensatory supination.

- **Pelvic Drop:** Angular inclination of the transverse hip vector relative to the horizontal baseline.

- **Dual-Path Asymmetry Classification:**
  * **Structural Asymmetry (Geometric):** Inter-limb differences in peak heel clearance during the swing phase (> 3%) highlight unilateral mobility or strength deficits.
  * **Temporal-Kinetic Asymmetry (Telemetric):** Symmetrical trajectories paired with uneven ground contact splits ($\vert{}50 - \text{GCT Balance}\vert{} > 2\%$) indicate protective limb unloading.


<br>


## 4. Biomechanical Benchmarks & Thresholds

Every calculated metric is assessed against clinical and athletic benchmarks to trigger diagnostic alerts:

| Metric (KPI) | Plane | Data Source | Target Range | Warning Threshold | Operational & Biomechanical Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cadence (SPM)** | Sagittal | Vision / FIT | 170 - 185 SPM | < 165 or > 195 SPM | Suboptimal stride length, energy expenditure |
| **Knee Strike Angle** | Sagittal | MediaPipe | 150° - 168° | > 170° (Overstride) | Braking impulse, patellofemoral overload |
| **Vertical Oscillation** | Sagittal | MediaPipe (cm) | 6.0 - 9.5 cm | > 10.0 cm | Excessive mechanical work against gravity |
| **Torso Lean** | Sagittal | MediaPipe | 5° - 10° | < 3° or > 12° | Poor gravitational load line, lumbar strain |
| **Leg Stiffness Index**| Sagittal | Sensor Fusion | 3.5 - 6.0 | < 3.5 (Low Stiffness)| Energy loss, prolonged stance phase |
| **Elbow Flexion** | Sagittal | MediaPipe | 80° - 95° | < 60° or > 105° | Upper kinetic chain tension, counter-torque issues |
| **Pronation / Supination** | Frontal | MediaPipe | -12° to +12° | < -12° or > +12° | Achilles tendon and plantar fascia strain |
| **Pelvic Drop** | Frontal | MediaPipe | < 4.0° | > 5.0° (Drop) | Gluteus medius functional insufficiency |
| **GCT Balance (L/R)** | Frontal | Garmin FIT | 49.0% - 51.0% | Deviation > 2.0% | Asymmetrical load attenuation |


<br>


## 5. Decision Intelligence & Actionable Insights

Rather than providing passive summary charts, the platform incorporates a prescriptive analytics rules engine that pairs anomalies with actionable corrective feedback:

```python
# Prescriptive diagnostic rule example (side_view_analysis.py)
if avg_l > 170:
    advice.append({
        "issue": "Overstriding (landing with an excessively straight knee)",
        "consequence": (
            "Landing far ahead of the center of mass generates braking forces"
            " and increases knee joint stress."
        ),
        "fix": "Shorten your stride and increase cadence by approximately 5%."
    })
```


<br>


## 6. Technology Stack
*Core Runtime: Python 3.10+
*Data Modeling & Signal Processing: Pandas, NumPy, SciPy
*ETL & Ingestion: fitparse (Garmin binary FIT protocol decoder), OpenCV
*Feature Extraction & Pose Estimation: MediaPipe Pose (Model Complexity 2)
*Web Layer & Visualization: Flask, Jinja2, HTML5/CSS, OpenCV HUD


<br>


## 7. Project Structure
```
fitform-gait-analytics/
├── static/                   # Output videos with HUD overlays and static assets
│   ├── .gitkeep
│   ├── result_running_rear.mp4
│   └── result_running_side.mp4
├── templates/
│   ├── index.html            # Upload interface (video, .FIT files, user height)
│   └── results.html          # Interactive biomechanical evaluation report
├── uploads/                  # Temporary ingestion cache
├── sample_data/              # Sample telemetry and input test videos
├── app.py                    # Flask server entrypoint and routing
├── rear_view_analysis.py     # Frontal plane pipeline (foot mechanics, pelvis)
├── side_view_analysis.py     # Sagittal plane pipeline (kinematics, sensor fusion)
├── requirements.txt          # Python dependencies
├── README.pl.md              # Polish documentation

```


<br>


## 8. Getting Started

Clone the repository:

```bash
git clone [https://github.com/laura-szczerbowska/GaitAnalytics.git](https://github.com/laura-szczerbowska/GaitAnalytics.git)
cd GaitAnalytics
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the development server:
```bash
python app.py
```
Open your browser and navigate to: http://127.0.0.1:5000


<br>


## 9. Future Roadmap
* Automated temporal offset synchronization (Auto-Sync Offset) between video streams and FIT telemetry based on initial acceleration peaks.
* GPU acceleration support (CUDA / TensorRT) for high-framerate 4K video extraction (60/120 FPS).
* Frontal view integration to assess dynamic knee valgus angles during deceleration.
* Automated export of diagnostic sessions to structured PDF reports with longitudinal progress tracking.


 <br>


## 10. Engineering Takeaways
* **Data Integrity & Filtering (Garbage In, Garbage Out)**: Raw landmark coordinates without thresholding exhibit high variance. FIFO buffers and percentile-based distance gating reduced kinematic variance by over 30%.
* **Value of Sensor Fusion**: Monocular video alone lacks millisecond-level ground strike accuracy, while wearable telemetry cannot assess spatial joint alignment. Fusing both data streams made composite metrics (e.g., Leg Stiffness Index) achievable.
* **Prescriptive Over Descriptive Delivery**: Translating low-level joint kinematic time series into explicit diagnostic corrections bridges the gap between raw data outputs and athletic decision-making.

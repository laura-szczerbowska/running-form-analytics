from collections import deque  # Fixed-size FIFO queues
import io
import os
import cv2  # Image and video processing
import fitparse  # Reading .FIT files (Garmin native data format)
import mediapipe as mp  # Human pose detection and tracking, landmarks
import numpy as np
import pandas as pd


# Helper functions
def smooth_value(history, new_val):  # Handle missing values
  if new_val is None or np.isnan(new_val) or new_val == 0:
    return (
        sum(history) / len(history) if history else 0
    )  # Average of historical values
  history.append(new_val)
  return sum(history) / len(history)


def get_tilt(p1, p2):
  dx, dy = p2[0] - p1[0], p2[1] - p1[1]
  return np.degrees(
      np.arctan2(dy, dx)
  )  # Convert via arctan2 (-180 to 180 deg, preserves direction) to degrees


# Foot tilt calculation
def get_tilt1(p1, p2):
  dx, dy = p2[0] - p1[0], p2[1] - p1[1]
  return np.degrees(np.arctan2(dy, dx)) + 90


# -------------------
def load_garmin_data(path):
  if not path or not os.path.exists(path):
    return None
  try:
    with open(path, "rb") as f:
      fit_data = f.read()
    fitfile = fitparse.FitFile(io.BytesIO(fit_data))
    records = []
    for record in fitfile.get_messages("record"):
      records.append({m.name: m.value for m in record})
    df = pd.DataFrame(records)
    if not df.empty and "timestamp" in df.columns:
      df["rel_time"] = (
          df["timestamp"] - df["timestamp"].iloc[0]
      ).dt.total_seconds()
      return df
  except Exception as e:
    print(f"Garmin error: {e}")
  return None


def run_back_analysis(VIDEO_PATH, USER_HEIGHT_CM, GARMIN_PATH=None):
  df_garmin = load_garmin_data(GARMIN_PATH)
  cap = cv2.VideoCapture(VIDEO_PATH)
  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  save_fps = fps / 2  # Divide FPS by 2 to write every second frame

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  out_size = (
      (800, int(800 * (height / width)))
      if width > 0 and height > 0
      else (800, 600)
  )

  output_filename = "result_" + os.path.basename(
      VIDEO_PATH
  )  # Allows serving the processed video on the web page
  output_path = os.path.join("static", output_filename)

  fourcc = cv2.VideoWriter_fourcc(*"avc1")
  out = cv2.VideoWriter(output_path, fourcc, save_fps, out_size)

  mp_pose = mp.solutions.pose  # Initialize pose tracking solution
  pose = mp_pose.Pose(
      min_detection_confidence=0.6,
      min_tracking_confidence=0.6,
      model_complexity=2,
  )

  pelvis_tilt_hist = deque(maxlen=30)  # FIFO queue
  hist_tilt, hist_time = [], []  # Pelvis data: angle, symmetry
  heels_L, heels_R = [], []  # Heel trajectory tracking
  hist_gct_balance = []  # Garmin ground contact time balance
  hist_dev_L = []  # Pronation / supination deviations
  hist_dev_R = []

  # Smoothing buffers to prevent stance phase angle jitter
  dev_L_hist = deque(maxlen=6)
  dev_R_hist = deque(maxlen=6)

  # FIFO history buffer to track lowest foot positions across frames
  lowest_y_history = deque(maxlen=40)

  # Assuming watch recording and video capture started at the same instant
  AUTO_OFFSET = 0
  frame_count = 0  # Frame skip counter

  while cap.isOpened():  # Process while video feed is active
    ret, frame = cap.read()
    if not ret:
      break

    frame_count += 1
    video_time = (
        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    )  # Convert milliseconds to seconds

    # Performance optimization: process every 2nd frame
    if frame_count % 2 == 0:
      display_frame = cv2.resize(frame, out_size)
      bh, bw, _ = display_frame.shape  # Resized frame dimensions

      g_gct_balance = 50.0  # Default to ideal symmetry
      if df_garmin is not None:
        target_sec = video_time - AUTO_OFFSET
        row = df_garmin.iloc[
            (df_garmin["rel_time"] - target_sec).abs().argsort()[:1]
        ]
        if not row.empty:
          g_gct_balance = row.get(
              "left_right_balance", pd.Series([50.0])
          ).values[0]

      # Default foot state in current frame
      foot_state_L, foot_state_R = "in flight", "in flight"
      deviation_L, deviation_R = None, None
      color_L, color_R = (120, 120, 120), (120, 120, 120)
      tilt = 0.0

      results = pose.process(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

      if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        hip_L, hip_R = [lm[23].x, lm[23].y], [lm[24].x, lm[24].y]  # Hips
        heel_L, heel_R = [lm[29].x, lm[29].y], [lm[30].x, lm[30].y]  # Heels
        ankle_L, ankle_R = [lm[27].x, lm[27].y], [lm[28].x, lm[28].y]  # Ankles

        # -----------------------
        # Adaptive ground level estimation
        frame_lowest = max(ankle_L[1], ankle_R[1], heel_L[1], heel_R[1])
        lowest_y_history.append(frame_lowest)

        # Baseline ground contact threshold from the 85th percentile of lowest vertical coordinates
        ground_baseline = np.percentile(lowest_y_history, 85)
        contact_margin = 0.045  # Vertical margin window for foot plant detection

        # Stance phase activation: foot must be within contact margin of the ground baseline
        is_contact_L = (
            max(ankle_L[1], heel_L[1]) >= (ground_baseline - contact_margin)
            and (ankle_L[1] - hip_L[1]) > (ankle_R[1] - hip_R[1] - 0.03)
        )
        is_contact_R = (
            max(ankle_R[1], heel_R[1]) >= (ground_baseline - contact_margin)
            and (ankle_R[1] - hip_R[1]) > (ankle_L[1] - hip_L[1] - 0.03)
        )

        # Biomechanical tolerance threshold (accounts for natural eversion and perspective)
        tolerance = 12.0

        # LEFT FOOT
        if lm[27].visibility > 0.6 and lm[29].visibility > 0.6 and is_contact_L:
          raw_tilt_L = get_tilt1(heel_L, ankle_L)
          deviation_L = smooth_value(dev_L_hist, raw_tilt_L)
          hist_dev_L.append(deviation_L)

          x_1, y_1 = int(heel_L[0] * bw), int(heel_L[1] * bh)
          x_2, y_2 = int(ankle_L[0] * bw), int(ankle_L[1] * bh)

          if deviation_L < -tolerance:
            foot_state_L = "pronation"
            color_L = (0, 0, 255)  # Red (BGR)
          elif deviation_L > tolerance:
            foot_state_L = "supination"
            color_L = (255, 0, 0)  # Blue (BGR)
          else:
            foot_state_L = "neutral"
            color_L = (0, 255, 0)  # Green (BGR)

          # Draw visual markers only during ground contact
          cv2.circle(display_frame, (x_1, y_1), 6, (255, 255, 0), -1)
          cv2.line(display_frame, (x_1, y_1), (x_2, y_2), color_L, 3)
          cv2.line(
              display_frame,
              (x_1 - 30, y_1),
              (x_1 + 30, y_1),
              (255, 255, 255),
              1,
          )
        else:
          dev_L_hist.clear()
          foot_state_L = "in flight"
          deviation_L = None
          color_L = (120, 120, 120)

        # RIGHT FOOT
        if lm[28].visibility > 0.6 and lm[30].visibility > 0.6 and is_contact_R:
          raw_tilt_R = get_tilt1(heel_R, ankle_R)
          deviation_R = smooth_value(dev_R_hist, raw_tilt_R)
          hist_dev_R.append(deviation_R)

          x_3, y_3 = int(heel_R[0] * bw), int(heel_R[1] * bh)
          x_4, y_4 = int(ankle_R[0] * bw), int(ankle_R[1] * bh)

          if deviation_R > tolerance:
            foot_state_R = "pronation"
            color_R = (0, 0, 255)
          elif deviation_R < -tolerance:
            foot_state_R = "supination"
            color_R = (255, 0, 0)
          else:
            foot_state_R = "neutral"
            color_R = (0, 255, 0)

          cv2.circle(display_frame, (x_3, y_3), 6, (255, 0, 255), -1)
          cv2.line(display_frame, (x_3, y_3), (x_4, y_4), color_R, 3)
          cv2.line(
              display_frame,
              (x_3 - 30, y_3),
              (x_3 + 30, y_3),
              (255, 255, 255),
              1,
          )
        else:
          dev_R_hist.clear()
          foot_state_R = "in flight"
          deviation_R = None
          color_R = (120, 120, 120)

        # -----------------------
        # Pelvis tilt
        tilt = smooth_value(pelvis_tilt_hist, get_tilt(hip_L, hip_R))
        hist_tilt.append(tilt)
        hist_time.append(video_time)
        hist_gct_balance.append(g_gct_balance)

        # Heel vertical trajectory tracking
        if lm[29].visibility > 0.5:
          heels_L.append(heel_L[1])
        if lm[30].visibility > 0.5:
          heels_R.append(heel_R[1])

        # Pelvis alignment bar
        color_pelvis = (
            (0, 255, 0) if 48.0 < g_gct_balance < 52.0 else (0, 0, 255)
        )
        cv2.line(
            display_frame,
            (int(hip_L[0] * bw), int(hip_L[1] * bh)),
            (int(hip_R[0] * bw), int(hip_R[1] * bh)),
            color_pelvis,
            3,
        )

      # HUD overlay panel
      cv2.rectangle(display_frame, (10, 10), (350, 150), (0, 0, 0), -1)
      font = cv2.FONT_HERSHEY_SIMPLEX
      scale = 0.5  # Uniform font scale
      thickness = 1  # Font stroke weight
      x_pos = 20  # Left margin

      # Ground contact balance
      color_bal = (0, 255, 0) if 48.0 < g_gct_balance < 52.0 else (0, 0, 255)
      cv2.putText(
          display_frame,
          f"Balance: {g_gct_balance}% left leg",
          (x_pos, 40),
          font,
          scale,
          color_bal,
          thickness,
      )

      # Pelvis tilt
      cv2.putText(
          display_frame,
          f"Pelvis Tilt: {round(tilt, 1)} deg",
          (x_pos, 70),
          font,
          scale,
          (255, 255, 255),
          thickness,
      )

      # Formatting foot feedback depending on contact vs flight phase
      txt_L = (
          f"L Foot: {foot_state_L} ({deviation_L:.1f} deg off vertical)"
          if deviation_L is not None
          else "L Foot: in flight"
      )
      txt_R = (
          f"R Foot: {foot_state_R} ({deviation_R:.1f} deg off vertical)"
          if deviation_R is not None
          else "R Foot: in flight"
      )

      # Left foot state
      cv2.putText(
          display_frame,
          txt_L,
          (x_pos, 100),
          font,
          scale,
          color_L,
          thickness,
      )

      # Right foot state
      cv2.putText(
          display_frame,
          txt_R,
          (x_pos, 130),
          font,
          scale,
          color_R,
          thickness,
      )

      out.write(display_frame)

  cap.release()
  out.release()
  cv2.destroyAllWindows()

  # Summary report generation
  advice = []
  avg_gct_bal = np.mean(hist_gct_balance) if hist_gct_balance else 50.0
  avg_pelvis_tilt = np.mean(hist_tilt) if hist_tilt else 0.0

  if hist_gct_balance and abs(50 - avg_gct_bal) > 2:
    max_h_L = (
        np.percentile(heels_L, 5) if heels_L else 1
    )  # Discard 5% extreme outliers
    max_h_R = np.percentile(heels_R, 5) if heels_R else 1
    visual_asymmetry = abs(max_h_L - max_h_R) * 100

    if visual_asymmetry > 3.0:
      advice.append({
          "issue": "Load Asymmetry (Structural Discrepancy)",
          "consequence": (
              "One leg absorbs significantly higher load impact on every"
              " landing."
          ),
          "fix": "Evaluate hip mobility and calf/ankle complex strength.",
      })
    else:
      advice.append({
          "issue": "Temporal Asymmetry (Ground Contact Time)",
          "consequence": (
              "Leg mechanics look visually balanced, but one foot stays planted"
              " longer."
          ),
          "fix": (
              "Incorporate rhythm and reactive push-off drills on the trailing"
              " foot."
          ),
      })

  # Pelvis tilt evaluation
  if abs(avg_pelvis_tilt) > 5.0:
    advice.append({
        "issue": (
            f"Noticeable Pelvic Drop/Tilt (Mean: {round(avg_pelvis_tilt, 1)}°)"
        ),
        "consequence": (
            "Can cause uneven load distribution along the lumbar spine and hip"
            " joints."
        ),
        "fix": (
            "Strengthen gluteus medius muscles (e.g., lateral leg raises) and"
            " deep core stabilizers."
        ),
    })

  # Pronation / supination aggregate assessment
  avg_dev_L = np.mean(hist_dev_L) if hist_dev_L else 0.0
  avg_dev_R = np.mean(hist_dev_R) if hist_dev_R else 0.0

  # Left foot diagnostic
  if avg_dev_L < -tolerance:
    advice.append({
        "issue": (
            f"Significant Left Foot Pronation (Mean: {round(avg_dev_L, 1)}°)"
        ),
        "consequence": (
            "Inward foot collapse elevates chronic strain on the Achilles"
            " tendon and knee joint."
        ),
        "fix": (
            "Use stability footwear or perform strengthening exercises for the"
            " plantar fascia and tibialis posterior."
        ),
    })
  elif avg_dev_L > tolerance:
    advice.append({
        "issue": (
            f"Significant Left Foot Supination (Mean: {round(avg_dev_L, 1)}°)"
        ),
        "consequence": (
            "Landing on the lateral edge diminishes shock absorption and"
            " increases lateral ankle sprain risk."
        ),
        "fix": (
            "Opt for cushioned footwear and focus on ankle joint mobility."
        ),
    })

  # Right foot diagnostic
  if avg_dev_R > tolerance:
    advice.append({
        "issue": (
            f"Significant Right Foot Pronation (Mean: {round(avg_dev_R, 1)}°)"
        ),
        "consequence": (
            "Inward collapse increases torsional stress on the knee joint and"
            " hip adductors."
        ),
        "fix": (
            "Strengthen intrinsic foot musculature and practice single-leg"
            " balance stability drills."
        ),
    })
  elif avg_dev_R < -tolerance:
    advice.append({
        "issue": (
            f"Significant Right Foot Supination (Mean: {round(avg_dev_R, 1)}°)"
        ),
        "consequence": (
            "Insufficient inward pronation transfers higher impact forces"
            " directly into joints."
        ),
        "fix": (
            "Improve iliotibial band (IT band) flexibility and verify adequate"
            " running shoe cushioning."
        ),
    })

  if not advice:
    advice.append({
        "issue": "No significant biomechanical issues detected",
        "consequence": (
            "Foot mechanics and pelvic alignment are within normal"
            " physiological thresholds."
        ),
        "fix": "Maintain current training program and loading schedule.",
    })

  return {
      "video": output_filename,
      "view_type": "back",
      "gct_balance": round(avg_gct_bal, 1),
      "pelvis_tilt": round(avg_pelvis_tilt, 1),
      "dev_left": round(avg_dev_L, 1),
      "dev_right": round(avg_dev_R, 1),
      "advice": advice,
  }
from collections import deque
import io
import os
import zipfile
import cv2
import fitparse
import mediapipe as mp
import numpy as np
import pandas as pd


# Helper functions
def smooth_value(history, new_val):
  if new_val is None or np.isnan(new_val) or new_val == 0:
    return sum(history) / len(history) if history else 0
  history.append(new_val)
  return sum(history) / len(history)


def calculate_angle(a, b, c):
  """Calculates the absolute internal angle at vertex B (0-180 deg) using the dot product."""
  a, b, c = np.array(a), np.array(b), np.array(c)
  ba = a - b
  bc = c - b
  cosine_angle = np.dot(ba, bc) / (
      np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
  )
  cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
  return float(np.degrees(np.arccos(cosine_angle)))


# Load Garmin telemetry (.FIT / .ZIP)
def load_garmin_data(path):
  if not path or not os.path.exists(path):
    return None
  try:
    if path.endswith(".zip"):
      with zipfile.ZipFile(path, "r") as z:
        fit_files = [f for f in z.namelist() if f.endswith(".fit")]
        if not fit_files:
          return None
        fit_data = z.read(fit_files[0])
    else:
      with open(path, "rb") as f:
        fit_data = f.read()

    fitfile = fitparse.FitFile(io.BytesIO(fit_data))
    records = [
        {m.name: m.value for m in record}
        for record in fitfile.get_messages("record")
    ]
    df = pd.DataFrame(records)

    if not df.empty and "timestamp" in df.columns:
      df["rel_time"] = (
          df["timestamp"] - df["timestamp"].iloc[0]
      ).dt.total_seconds()
      return df
  except Exception as e:
    print(f"Garmin error: {e}")
  return None


# Main side-view biomechanical analysis pipeline
def run_side_analysis(
    VIDEO_PATH, USER_HEIGHT_CM, GARMIN_PATH=None, view_type="side"
):
  df_garmin = load_garmin_data(GARMIN_PATH)
  cap = cv2.VideoCapture(VIDEO_PATH)
  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  save_fps = fps

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  out_size = (
      (800, int(800 * (height / width)))
      if width > 0 and height > 0
      else (800, 600)
  )

  output_filename = "result_" + os.path.basename(VIDEO_PATH)
  output_path = os.path.join("static", output_filename)

  fourcc = cv2.VideoWriter_fourcc(*"avc1")
  out = cv2.VideoWriter(output_path, fourcc, save_fps, out_size)

  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(
      min_detection_confidence=0.6,
      min_tracking_confidence=0.6,
      model_complexity=2,
  )

  # FIFO smoothing buffers
  ankle_x_history = deque(maxlen=7)  # Buffer to detect foot touchdown moment
  ankle_y_history = deque(maxlen=7)
  stiffness_history = deque(maxlen=20)
  hip_y_history = deque(maxlen=40)
  osc_history = deque(maxlen=20)
  knee_angle_history = deque(maxlen=5)  # Shortened buffer for dynamic knee angle response
  tilt_history = deque(maxlen=30)
  cadence_history = deque(maxlen=15)
  step_timestamps = deque(maxlen=10)
  arm_angle_history = deque(maxlen=10)

  # Body side lock (prevents switching limbs mid-flight)
  dominant_side = None
  side_votes = {"right": 0, "left": 0}

  # Storage for session summary statistics
  hist_cad_ai, hist_osc_ai, hist_land, hist_stiff, hist_arm = [], [], [], [], []
  hist_tilt = []
  hist_gct = []

  # State variables
  stage = "air"
  last_landing_angle = 0
  min_knee_angle = 180
  cadence, osc_ai_cm, current_tilt, current_stiff, current_arm = 0, 0, 0, 0, 0
  g_cad, g_gct = 0, 0

  AUTO_OFFSET = 0
  frame_count = 0

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      frame_count += 1
      video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

      display_frame = cv2.resize(frame, out_size)
      bh, bw, _ = display_frame.shape

      # Synchronize Garmin telemetry
      if df_garmin is not None:
          target_sec = video_time - AUTO_OFFSET
          row = df_garmin.iloc[
              (df_garmin["rel_time"] - target_sec).abs().argsort()[:1]
          ]
          if not row.empty:
              raw_g_cad = row.get("cadence", 0)
              g_cad = (
                  int(raw_g_cad.values[0] * 2)
                  if pd.notnull(raw_g_cad).any()
                  else 0
              )
              g_gct = row.get("stance_time", pd.Series([0])).values[0]
              if pd.notnull(g_gct) and g_gct > 0:
                  hist_gct.append(g_gct)

      results = pose.process(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

      if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Keypoints for both legs: Left (23, 25, 27) and Right (24, 26, 28)
        l_hip = [lm[23].x, lm[23].y]
        l_knee = [lm[25].x, lm[25].y]
        l_ankle = [lm[27].x, lm[27].y]

        r_hip = [lm[24].x, lm[24].y]
        r_knee = [lm[26].x, lm[26].y]
        r_ankle = [lm[28].x, lm[28].y]

        # Keypoints for both arms: Left (11, 13, 15) and Right (12, 14, 16)
        l_shldr = [lm[11].x, lm[11].y]
        l_elbow = [lm[13].x, lm[13].y]
        l_wrist = [lm[15].x, lm[15].y]

        r_shldr = [lm[12].x, lm[12].y]
        r_elbow = [lm[14].x, lm[14].y]
        r_wrist = [lm[16].x, lm[16].y]

        # Determine and permanently lock the dominant side at the start of the video
        if dominant_side is None:
            r_score = lm[24].visibility + lm[26].visibility + lm[28].visibility
            l_score = lm[23].visibility + lm[25].visibility + lm[27].visibility
            if r_score >= l_score:
                side_votes["right"] += 1
            else:
                side_votes["left"] += 1

            if (side_votes["right"] + side_votes["left"]) >= 25:
                dominant_side = "right" if side_votes["right"] >= side_votes["left"] else "left"
            active_side = "right" if r_score >= l_score else "left"
        else:
            active_side = dominant_side

        # Hard-assign joints to the selected dominant side
        if active_side == "right":
            shldr, elbow, wrist = r_shldr, r_elbow, r_wrist
            hip, knee, ankle = r_hip, r_knee, r_ankle
        else:
            shldr, elbow, wrist = l_shldr, l_elbow, l_wrist
            hip, knee, ankle = l_hip, l_knee, l_ankle

        # Vertical oscillation
        hip_y_history.append(hip[1])
        torso_px = np.linalg.norm(np.array(shldr) - np.array(hip))
        if torso_px > 0:
          px_to_cm = (USER_HEIGHT_CM * 0.3) / torso_px
          raw_osc_px = np.percentile(hip_y_history, 95) - np.percentile(
              hip_y_history, 5
          )
          osc_ai_cm = round(smooth_value(osc_history, raw_osc_px * px_to_cm), 1)

        # Raw and smoothed knee angle in pixel coordinates
        knee_pt_a = [hip[0] * bw, hip[1] * bh]
        knee_pt_b = [knee[0] * bw, knee[1] * bh]
        knee_pt_c = [ankle[0] * bw, ankle[1] * bh]
        raw_knee_angle = calculate_angle(knee_pt_a, knee_pt_b, knee_pt_c)
        current_knee = round(smooth_value(knee_angle_history, raw_knee_angle), 0)

        raw_tilt = np.degrees(
            np.arctan2(shldr[0] - hip[0], -(shldr[1] - hip[1]))
        )
        if -15 < raw_tilt < 25:
          current_tilt = int(smooth_value(tilt_history, raw_tilt))

        # Elbow flexion angle
        arm_pt_a = [shldr[0] * bw, shldr[1] * bh]
        arm_pt_b = [elbow[0] * bw, elbow[1] * bh]
        arm_pt_c = [wrist[0] * bw, wrist[1] * bh]
        raw_arm_angle = calculate_angle(arm_pt_a, arm_pt_b, arm_pt_c)
        current_arm = int(smooth_value(arm_angle_history, raw_arm_angle))

        # Touchdown detection based on maximum forward ankle extension (end of flight phase)
        ankle_x_history.append(ankle[0])
        ankle_y_history.append(ankle[1])

        # Initial Contact
        if len(ankle_x_history) >= 3 and stage == "air":
          # Foot stops moving forward (starts being pulled backward by treadmill belt):
          foot_touchdown = (
              ankle_x_history[-2] >= ankle_x_history[-1]
              and ankle_x_history[-2] > ankle_x_history[-3]
          )

          if foot_touchdown:
            stage = "contact"
            # Capture instantaneous flexion angle at the moment of touchdown
            last_landing_angle = round(raw_knee_angle, 0)
            min_knee_angle = raw_knee_angle
            step_timestamps.append(video_time)

            if len(step_timestamps) > 1:
              dt = step_timestamps[-1] - step_timestamps[-2]
              if dt > 0.3:
                raw_spm = 120 / dt
                if 110 < raw_spm < 230:
                  cadence = int(smooth_value(cadence_history, raw_spm))

        # Track maximum flexion during the stance phase
        if stage == "contact" and raw_knee_angle < min_knee_angle:
          min_knee_angle = raw_knee_angle

        # Transition to flight phase (foot lifts after push-off)
        if stage == "contact" and len(ankle_y_history) >= 2:
          if ankle_y_history[-1] < ankle_y_history[-2] - 0.015:
            stage = "air"
            flexion = 180 - min_knee_angle

            gct_val = g_gct if (pd.notnull(g_gct) and g_gct > 0) else 198.0
            if flexion > 5:
              # Constant 50000 adjusted to maintain physiological stiffness scaling
              current_stiff = round(50000 / (gct_val * flexion), 2)
              stiffness_history.append(current_stiff)

        # Append metrics for aggregate reporting
        hist_cad_ai.append(cadence)
        hist_osc_ai.append(osc_ai_cm)
        hist_land.append(last_landing_angle)
        hist_stiff.append(current_stiff)
        hist_tilt.append(current_tilt)
        hist_arm.append(current_arm)

        # Draw skeletal links and anatomical joints
        joints = [shldr, hip, knee, ankle, elbow, wrist]
        connections = [
            [shldr, hip],
            [hip, knee],
            [knee, ankle],
            [shldr, elbow],
            [elbow, wrist],
        ]

        for p1, p2 in connections:
          pt1 = (int(p1[0] * bw), int(p1[1] * bh))
          pt2 = (int(p2[0] * bw), int(p2[1] * bh))
          cv2.line(display_frame, pt1, pt2, (255, 255, 255), 2)

        for pt in joints:
          coord = (int(pt[0] * bw), int(pt[1] * bh))
          cv2.circle(display_frame, coord, 4, (0, 255, 255), -1)

      # On-Screen HUD Overlay
      cv2.rectangle(display_frame, (10, 10), (330, 260), (0, 0, 0), -1)

      font = cv2.FONT_HERSHEY_SIMPLEX
      scale = 0.45
      thickness = 1
      x_pos = 20

      disp_cad = cadence if cadence > 0 else g_cad
      cv2.putText(
          display_frame,
          f"AI Cadence: {disp_cad} SPM | (Garmin: {g_cad})",
          (x_pos, 40),
          font,
          scale,
          (255, 255, 0),
          thickness,
      )
      cv2.putText(
          display_frame,
          f"Vertical Oscillation: {osc_ai_cm} cm",
          (x_pos, 80),
          font,
          scale,
          (255, 255, 255),
          thickness,
      )
      cv2.putText(
          display_frame,
          f"Torso Lean: {current_tilt} deg",
          (x_pos, 120),
          font,
          scale,
          (255, 255, 255),
          thickness,
      )
      cv2.putText(
          display_frame,
          f"Knee Flexion (Landing): {int(last_landing_angle)} deg",
          (x_pos, 160),
          font,
          scale,
          (0, 255, 0),
          thickness,
      )
      cv2.putText(
          display_frame,
          f"Leg Stiffness: {current_stiff}",
          (x_pos, 200),
          font,
          scale,
          (255, 0, 255),
          thickness,
      )
      cv2.putText(
          display_frame,
          f"Arm Flexion Angle: {current_arm} deg",
          (x_pos, 240),
          font,
          scale,
          (255, 0, 255),
          thickness,
      )

      out.write(display_frame)

  cap.release()
  out.release()
  cv2.destroyAllWindows()

  # Final summary report calculation
  if hist_land:
    valid_cad = [x for x in hist_cad_ai if x > 0]
    if valid_cad:
      avg_c = np.mean(valid_cad)
    elif df_garmin is not None and "cadence" in df_garmin.columns:
      garmin_cads = df_garmin["cadence"].dropna() * 2
      avg_c = garmin_cads[garmin_cads > 0].mean() if not garmin_cads.empty else g_cad
    else:
      avg_c = g_cad if g_cad > 0 else 0

    avg_o = (
        np.mean([x for x in hist_osc_ai if x > 0])
        if any(np.array(hist_osc_ai) > 0)
        else 0
    )
    avg_l = (
        np.mean([x for x in hist_land if x > 0])
        if any(np.array(hist_land) > 0)
        else 0
    )
    avg_s = (
        np.mean([x for x in hist_stiff if x > 0])
        if any(np.array(hist_stiff) > 0)
        else 0
    )
    avg_tilt = (
        np.mean([x for x in hist_tilt if not np.isnan(x)])
        if hist_tilt
        else 0
    )
    avg_arm = (
        np.mean([x for x in hist_arm if x > 0])
        if any(np.array(hist_arm) > 0)
        else 0
    )

    advice = []
    if avg_l > 170:
      advice.append({
          "issue": "Overstriding (landing with an excessively straight knee)",
          "consequence": (
              "Landing far ahead of the center of mass generates braking forces"
              " and increases knee joint stress."
          ),
          "fix": "Shorten your stride and increase cadence by approximately 5%.",
      })
    if avg_o > 10:
      advice.append({
          "issue": "Excessive vertical oscillation",
          "consequence": (
              "Decreases running economy, accelerates fatigue, and causes"
              " higher landing impact."
          ),
          "fix": (
              "Direct your force vectors forward; visualize running through a"
              " low-ceiling tunnel."
          ),
      })
    if 0 < avg_s < 3.5:
      advice.append({
          "issue": "Low leg stiffness",
          "consequence": (
              "Reduces elastic energy return and extends ground contact time."
          ),
          "fix": (
              "Incorporate dynamic skips, bounding, and jump rope drills to"
              " build reactive tendon strength."
          ),
      })
    if avg_tilt < 3:
        advice.append({
            "issue": "Upright or backward torso lean (<3 deg)",
            "consequence": (
                "Promotes landing ahead of the center of mass, increasing braking"
                " forces and preventing the effective use of gravity for forward"
                " propulsion."
            ),
            "fix": (
                "Initiate a slight whole-body forward lean (approx. 5 deg-8 deg)"
                " originating from the ankles, rather than bending at the waist."
            ),
        })
    elif avg_tilt > 12:
        advice.append({
            "issue": "Excessive forward torso lean (>12 deg)",
            "consequence": (
                "Often stems from bending at the hips (hip collapse), which"
                " strains the lower back and restricts full hip extension."
            ),
            "fix": (
                "Engage your core, keep your chest upright, and direct your gaze"
                " 10-15 meters forward rather than looking straight down."
            ),
        })

    if 0 < avg_c < 165:
        advice.append({
            "issue": "Low running cadence (<165 SPM)",
            "consequence": (
                "Leads to higher vertical impact loads per step and often"
                " correlates with overstriding."
            ),
            "fix": (
                "Aim to increase your step frequency towards 170-180 SPM by taking"
                " quicker, shorter steps (e.g., using a running metronome)."
            ),
        })
    elif avg_c > 195:
        advice.append({
            "issue": "Very high running cadence (>195 SPM)",
            "consequence": (
                "May limit full hip extension and reduce propulsion efficiency at"
                " aerobic paces."
            ),
            "fix": (
                "Allow a slightly longer, natural push-off while maintaining"
                " compact foot plant."
            ),
        })

    if avg_arm > 105:
      advice.append({
          "issue": "Excessively extended arms (> 105 deg)",
          "consequence": (
              "Increases shoulder girdle fatigue and hinders cadence"
              " maintenance."
          ),
          "fix": "Keep elbows bent closer to approximately 90°.",
      })
    elif 0 < avg_arm < 60:
      advice.append({
          "issue": "Arms held too tightly (<60 deg)",
          "consequence": "Causes upper back and neck tension.",
          "fix": "Relax your shoulders and let elbows swing freely within 80 deg-90 deg.",
      })

    if not advice:
      advice.append({
          "issue": "Running Technique Assessment",
          "consequence": "Optimal biomechanics",
          "fix": (
              "No significant biomechanical deviations detected. Running form"
              " is stable and efficient."
          ),
      })

    return {
        "video": output_filename,
        "view_type": "side",
        "cadence": round(avg_c, 1),
        "stiffness": round(avg_s, 2),
        "arm_angle": int(avg_arm),
        "oscillation": round(avg_o, 1),
        "landing_angle": int(avg_l),
        "torso_lean": round(avg_tilt, 1),
        "advice": advice,
    }

  return {
      "video": output_filename,
      "view_type": "side",
      "cadence": 0,
      "stiffness": 0,
      "arm_angle": 0,
      "oscillation": 0,
      "landing_angle": 0,
      "torso_lean": 0,
      "advice": [],
  }
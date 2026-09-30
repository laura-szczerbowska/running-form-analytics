import os
from flask import Flask, render_template, request
from rear_view_analysis import run_back_analysis
from side_view_analysis import run_side_analysis

app = Flask(__name__)

# File storage configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure required directories exist
for folder in [UPLOAD_FOLDER, STATIC_FOLDER]:
  if not os.path.exists(folder):
    os.makedirs(folder)


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
  video = request.files.get('video')
  garmin_file = request.files.get('garmin_file')
  view_type = request.form.get('view_type', 'side')
  user_height_raw = request.form.get('user_height')

  # Validate biometric scaling input
  try:
    user_height = float(user_height_raw)
  except (ValueError, TypeError):
    return 'Error: Please provide a valid height in cm (e.g., 175 or 182)!'

  if video and video.filename != '':
    # Persist uploaded media assets
    v_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(v_path)

    g_path = None
    if garmin_file and garmin_file.filename != '':
      g_path = os.path.join(app.config['UPLOAD_FOLDER'], garmin_file.filename)
      garmin_file.save(g_path)

    # Route execution pipeline by camera view
    if view_type == 'back':
      results = run_back_analysis(
          v_path, USER_HEIGHT_CM=user_height, GARMIN_PATH=g_path
      )
    else:
      results = run_side_analysis(
          v_path, USER_HEIGHT_CM=user_height, GARMIN_PATH=g_path
      )

    return render_template('results.html', results=results)

  return 'Error: No video file selected!'


if __name__ == '__main__':
  app.run(debug=True)
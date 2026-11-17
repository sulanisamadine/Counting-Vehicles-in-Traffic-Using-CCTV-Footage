from flask import Flask, render_template, request, jsonify
import os
import cv2
import base64
import subprocess
import traceback
from counter import VehicleCounter
from processor import DIPProcessor
from dip_evidence import DIPEvidence

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

def extract_sample_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("❌ Could not read sample frame")
            return None

        processor = DIPProcessor()
        original  = frame.copy()
        noise     = processor.apply_noise_removal(original.copy())
        sharp     = processor.apply_sharpening(noise.copy())
        clahe     = processor.apply_contrast_enhancement(sharp.copy())
        bright    = processor.apply_brightness_normalization(clahe.copy())
        morph     = processor.apply_morphological_operations(bright.copy())

        return {
            'original':  frame_to_base64(original),
            'noise':     frame_to_base64(noise),
            'sharp':     frame_to_base64(sharp),
            'clahe':     frame_to_base64(clahe),
            'bright':    frame_to_base64(bright),
            'morph':     frame_to_base64(morph),
            'processed': frame_to_base64(morph),
            'final':     frame_to_base64(morph)
        }
    except Exception as e:
        print(f"❌ Frame extraction error: {e}")
        traceback.print_exc()
        return None

def convert_to_web_mp4(input_path, output_path):
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and os.path.exists(output_path):
            print("✅ ffmpeg conversion successful")
            return True
        else:
            print(f"❌ ffmpeg failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        print("📥 Received video upload request")

        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        print(f"📁 Processing file: {file.filename}")

        # Clean filename - remove special characters
        safe_name    = file.filename.replace(' ', '_')
        input_path   = os.path.join(UPLOAD_FOLDER, safe_name)
        base_name    = safe_name.rsplit('.', 1)[0]
        raw_output   = os.path.join(OUTPUT_FOLDER, f'raw_{base_name}.mp4')
        final_output = os.path.join(OUTPUT_FOLDER, f'final_{base_name}.mp4')

        file.save(input_path)
        print(f"✅ File saved to: {input_path}")

        # Extract DIP frames
        print("📸 Extracting DIP sample frames...")
        frames = extract_sample_frames(input_path)
        print(f"✅ Frames extracted: {frames is not None}")

        # Process video
        print("🚗 Starting vehicle detection...")
        counter = VehicleCounter()
        counts, total = counter.process_video(input_path, raw_output)
        print(f"✅ Detection complete. Total: {total}, Counts: {counts}")

        # Convert video
        print("🎬 Converting video for browser...")
        converted = convert_to_web_mp4(raw_output, final_output)

        if converted and os.path.exists(final_output):
            video_url = '/static/output/' + f'final_{base_name}.mp4'
        elif os.path.exists(raw_output):
            video_url = '/static/output/' + f'raw_{base_name}.mp4'
        else:
            video_url = ''

        print(f"✅ Video URL: {video_url}")

        response_data = {
            'counts':       counts,
            'total':        total,
            'output_video': video_url,
            'frames':       frames
        }

        print(f"✅ Sending response successfully")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ ERROR in /process route:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/dip-evidence', methods=['POST'])
def dip_evidence():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        file      = request.files['video']
        safe_name = file.filename.replace(' ', '_')
        input_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.save(input_path)

        evidence  = DIPEvidence()
        evidence.generate_evidence(input_path)

        return jsonify({
            'original':    '/static/dip_evidence/original.jpg',
            'noise':       '/static/dip_evidence/noise.jpg',
            'sharp':       '/static/dip_evidence/sharp.jpg',
            'clahe':       '/static/dip_evidence/clahe.jpg',
            'bright':      '/static/dip_evidence/bright.jpg',
            'morph':       '/static/dip_evidence/morph.jpg',
            'final':       '/static/dip_evidence/final.jpg',
            'comparison':  '/static/dip_evidence/comparison.jpg',
            'all_steps':   '/static/dip_evidence/all_steps.jpg',
            'histogram':   '/static/dip_evidence/histogram.png'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
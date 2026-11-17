import cv2
import numpy as np
import os
from processor import DIPProcessor

class DIPEvidence:
    """
    Generates visual evidence of all DIP techniques applied.
    CS31092 - Digital Image Processing | Group 40
    """

    def __init__(self):
        self.processor = DIPProcessor()
        self.output_dir = 'static/dip_evidence'
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_frame(self, video_path, frame_number=None):
        """Extract a clear frame from the video"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frame from 1/3 of video (usually clearest traffic)
        if frame_number is None:
            frame_number = total // 3

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception("Could not read frame from video")
        return frame

    def add_label(self, frame, title, subtitle=""):
        """Add a professional label to a frame"""
        labeled = frame.copy()
        h, w = labeled.shape[:2]

        # Top banner
        cv2.rectangle(labeled, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(labeled, title,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        if subtitle:
            cv2.putText(labeled, subtitle,
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (180, 180, 180), 1)

        # Bottom banner
        cv2.rectangle(labeled, (0, h-25), (w, h), (0, 0, 0), -1)
        cv2.putText(labeled,
                    "CS31092 - DIP | Group 40 | KDU",
                    (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (100, 100, 100), 1)
        return labeled

    def generate_evidence(self, video_path):
        """Generate all DIP evidence images"""
        print("📸 Extracting frame from video...")
        original = self.extract_frame(video_path)

        # Apply each technique step by step
        print("🔬 Applying DIP techniques...")
        noise  = self.processor.apply_noise_removal(original.copy())
        sharp  = self.processor.apply_sharpening(noise.copy())
        clahe  = self.processor.apply_contrast_enhancement(sharp.copy())
        bright = self.processor.apply_brightness_normalization(clahe.copy())
        morph  = self.processor.apply_morphological_operations(bright.copy())

        frames = {
            'original': original,
            'noise':    noise,
            'sharp':    sharp,
            'clahe':    clahe,
            'bright':   bright,
            'morph':    morph,
            'final':    morph
        }

        # Save individual labeled frames
        labels = {
            'original': ('ORIGINAL - Raw CCTV Frame',
                         'No processing applied'),
            'noise':    ('STEP 1: Noise Removal',
                         'Gaussian Blur (5x5) + Median Blur (3x3)'),
            'sharp':    ('STEP 2: Image Sharpening',
                         'Kernel Filter: [[0,-1,0],[-1,5,-1],[0,-1,0]]'),
            'clahe':    ('STEP 3: Contrast Enhancement (CLAHE)',
                         'Clip Limit=3.0, Grid Size=(8x8), LAB Color Space'),
            'bright':   ('STEP 4: Brightness Normalization',
                         'Histogram Equalization on HSV V-channel'),
            'morph':    ('STEP 5: Morphological Operations',
                         'MORPH_OPEN + MORPH_CLOSE, Kernel 3x3'),
            'final':    ('FINAL RESULT - All DIP Techniques Applied',
                         'Ready for YOLO vehicle detection')
        }

        saved_paths = {}
        for key, frame in frames.items():
            title, subtitle = labels[key]
            labeled = self.add_label(frame, title, subtitle)
            path = os.path.join(self.output_dir, f'{key}.jpg')
            cv2.imwrite(path, labeled)
            saved_paths[key] = path
            print(f"✅ Saved: {path}")

        # Generate comparison images
        self._save_side_by_side(original, morph,
                                "ORIGINAL vs FINAL (All DIP Applied)")
        self._save_all_steps(frames)
        self._save_histogram_comparison(original, morph)

        print("✅ All DIP evidence generated!")
        return saved_paths

    def _save_side_by_side(self, frame1, frame2, title):
        """Save before/after side by side"""
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        # Resize to same height
        target_h = min(h1, h2)
        target_w = int(w1 * target_h / h1)

        f1 = cv2.resize(frame1, (target_w, target_h))
        f2 = cv2.resize(frame2, (target_w, target_h))

        # Add labels on each frame
        cv2.rectangle(f1, (0, 0), (target_w, 40), (200, 0, 0), -1)
        cv2.putText(f1, "BEFORE (Original)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        cv2.rectangle(f2, (0, 0), (target_w, 40), (0, 150, 0), -1)
        cv2.putText(f2, "AFTER (All DIP Applied)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

        # Combine side by side with divider
        divider = np.zeros((target_h, 10, 3), dtype=np.uint8)
        divider[:] = (0, 255, 255)
        combined = np.hstack([f1, divider, f2])

        # Title bar
        title_bar = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, title,
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)
        final = np.vstack([title_bar, combined])

        path = os.path.join(self.output_dir, 'comparison.jpg')
        cv2.imwrite(path, final)
        print(f"✅ Saved comparison: {path}")

    def _save_all_steps(self, frames):
        """Save all 6 steps in one grid image"""
        keys    = ['original', 'noise', 'sharp', 'clahe', 'bright', 'morph']
        titles  = ['1.Original', '2.Noise Removal',
                   '3.Sharpening', '4.CLAHE',
                   '5.Brightness', '6.Morphological']

        target_w = 400
        target_h = 300
        cols = 3
        rows = 2

        grid_w = cols * target_w
        grid_h = rows * target_h
        grid   = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx, (key, title) in enumerate(zip(keys, titles)):
            row = idx // cols
            col = idx % cols
            x   = col * target_w
            y   = row * target_h

            frame   = cv2.resize(frames[key], (target_w, target_h))
            # Add step label
            cv2.rectangle(frame, (0, 0), (target_w, 35), (0, 0, 0), -1)
            color = (0, 255, 255) if idx == 0 else (0, 255, 0)
            cv2.putText(frame, title,
                        (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
            grid[y:y+target_h, x:x+target_w] = frame

        # Add grid lines
        for i in range(1, cols):
            cv2.line(grid, (i*target_w, 0),
                     (i*target_w, grid_h), (0, 255, 255), 2)
        for i in range(1, rows):
            cv2.line(grid, (0, i*target_h),
                     (grid_w, i*target_h), (0, 255, 255), 2)

        # Title
        title_bar = np.zeros((60, grid_w, 3), dtype=np.uint8)
        cv2.putText(title_bar,
                    "ALL DIP TECHNIQUES - CS31092 | Group 40",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)
        final = np.vstack([title_bar, grid])

        path = os.path.join(self.output_dir, 'all_steps.jpg')
        cv2.imwrite(path, final)
        print(f"✅ Saved all steps grid: {path}")

    def _save_histogram_comparison(self, original, processed):
        """Save histogram comparison showing contrast improvement"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a2e')

        titles = ['Original - Blue Channel',
                  'Original - Green Channel',
                  'Processed - Blue Channel',
                  'Processed - Green Channel']
        imgs   = [original, original, processed, processed]
        chans  = [0, 1, 0, 1]
        colors = ['blue', 'green', 'blue', 'green']

        for ax, img, ch, color, title in zip(
                axes.flatten(), imgs, chans, colors, titles):
            hist = cv2.calcHist([img], [ch], None, [256], [0, 256])
            ax.set_facecolor('#0d1117')
            ax.plot(hist, color=color, linewidth=1.5)
            ax.set_title(title, color='white', fontsize=10)
            ax.set_xlim([0, 256])
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        fig.suptitle(
            'Histogram Comparison: Before vs After DIP Processing\n'
            'CS31092 - Digital Image Processing | Group 40',
            color='cyan', fontsize=13, fontweight='bold')

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'histogram.png')
        plt.savefig(path, facecolor='#1a1a2e', dpi=100)
        plt.close()
        print(f"✅ Saved histogram: {path}")
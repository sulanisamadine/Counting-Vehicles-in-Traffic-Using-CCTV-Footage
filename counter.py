import cv2
import numpy as np
from ultralytics import YOLO
from processor import DIPProcessor


class VehicleCounter:

    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.processor = DIPProcessor()
        self.vehicle_classes = {
            2: 'Car',
            3: 'Motorcycle',
            5: 'Bus',
            7: 'Truck'
        }
        self.counted_ids = {}          # tid -> last cy (for direction tracking)
        self.counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
        self.total = 0
        self.track_history = {}        # tid -> list of cy positions

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {input_path}")
            return self.counts, self.total

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        # Counting line at 55% height — more vehicles visible above the line
        line_position = int(height * 0.55)

        # Counting band — wider band catches fast-moving vehicles
        # Vehicles must cross from above (cy < line) to below (cy > line) or vice versa
        band_half = int(height * 0.04)   # ±4% of frame height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # FIX 1: Do NOT skip frames — this causes missed crossings.
            # Instead, run YOLO on every frame for accurate tracking.
            # If performance is needed, reduce model size (yolov8n) or resolution.

            # FIX 2: Preprocess a copy for YOLO — but draw on the original frame.
            # Running tracking on heavily filtered frames can reduce accuracy,
            # so we use the processed frame only for detection input.
            processed = self.processor.preprocess(frame.copy())

            try:
                results = self.model.track(
                    processed,
                    persist=True,
                    verbose=False,
                    conf=0.35,    # slightly higher conf = fewer false positives
                    iou=0.45,
                    tracker="botsort.yaml",   # BotSORT is more stable than default ByteTrack for dense traffic
                )
            except Exception as e:
                # Fallback: try default tracker
                try:
                    results = self.model.track(
                        processed,
                        persist=True,
                        verbose=False,
                        conf=0.35,
                        iou=0.45,
                    )
                except Exception as e2:
                    print(f"YOLO error: {e2}")
                    out.write(frame)
                    continue

            # ── Draw counting line ──────────────────────────────────────────
            # Main line
            cv2.line(frame,
                     (0, line_position),
                     (width, line_position),
                     (0, 255, 0), 2)

            # Band zone (shaded strip showing where counting happens)
            band_overlay = frame.copy()
            cv2.rectangle(band_overlay,
                          (0, line_position - band_half),
                          (width, line_position + band_half),
                          (0, 255, 0), -1)
            cv2.addWeighted(band_overlay, 0.08, frame, 0.92, 0, frame)

            # Label on the line
            lbl_w = 170
            cv2.rectangle(frame,
                          (width // 2 - lbl_w // 2, line_position - 24),
                          (width // 2 + lbl_w // 2, line_position + 6),
                          (0, 0, 0), -1)
            cv2.putText(frame, "[ COUNTING LINE ]",
                        (width // 2 - lbl_w // 2 + 5, line_position - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 2)

            # ── DIP label at bottom ─────────────────────────────────────────
            cv2.putText(frame,
                        "DIP: Gaussian+Median | Sharpen | CLAHE | BrightNorm | Morph",
                        (5, height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 200, 255), 1)

            # ── Process detections ──────────────────────────────────────────
            if (results and
                    results[0].boxes is not None and
                    len(results[0].boxes) > 0):

                boxes   = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()

                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    ids = list(range(len(boxes)))

                for i, (box, cls, conf) in enumerate(
                        zip(boxes, classes, confs)):

                    if cls not in self.vehicle_classes:
                        continue
                    if conf < 0.35:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx    = (x1 + x2) // 2
                    cy    = (y1 + y2) // 2
                    label = self.vehicle_classes[cls]
                    tid   = int(ids[i]) if i < len(ids) else i

                    # ── Track history for direction ─────────────────────────
                    if tid not in self.track_history:
                        self.track_history[tid] = []
                    self.track_history[tid].append(cy)
                    # Keep only last 8 positions
                    if len(self.track_history[tid]) > 8:
                        self.track_history[tid].pop(0)

                    # ── FIX 3: Direction-aware crossing detection ───────────
                    # A vehicle is counted only when its centre crosses the line
                    # from one side to the other (not just when it's near the line).
                    history = self.track_history[tid]
                    crossed = False
                    if tid not in self.counted_ids and len(history) >= 2:
                        prev_cy = history[-2]
                        curr_cy = history[-1]
                        # Crossed downward (top → bottom)
                        if prev_cy < line_position <= curr_cy:
                            crossed = True
                        # Crossed upward (bottom → top)
                        elif prev_cy > line_position >= curr_cy:
                            crossed = True

                    if crossed:
                        self.counted_ids[tid] = cy
                        self.counts[label] += 1
                        self.total += 1
                        print(f"✅ Counted {label} ID:{tid} "
                              f"| Total: {self.total}")

                    # ── Draw bounding box ───────────────────────────────────
                    color = {
                        'Car':        (0, 255, 0),
                        'Motorcycle': (0, 165, 255),
                        'Bus':        (255, 0, 255),
                        'Truck':      (255, 165, 0),
                    }.get(label, (200, 200, 200))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label background so text is always readable
                    lbl_text = f"{label} {conf:.2f} #{tid}"
                    (tw, th), _ = cv2.getTextSize(
                        lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame,
                                  (x1, y1 - th - 8),
                                  (x1 + tw + 4, y1),
                                  color, -1)
                    cv2.putText(frame, lbl_text,
                                (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 0), 1)

                    # Centre dot
                    dot_color = (0, 0, 255) if tid in self.counted_ids else (255, 255, 0)
                    cv2.circle(frame, (cx, cy), 5, dot_color, -1)

            # ── Draw count panel (TOP-RIGHT so it never overlaps anything) ──
            self._draw_counts(frame)
            out.write(frame)

        cap.release()
        out.release()
        print(f"\n✅ Done! Total: {self.total} | Counts: {self.counts}")
        return self.counts, self.total

    def _draw_counts(self, frame):
        """Draws the count panel in the TOP-RIGHT corner."""
        h, w = frame.shape[:2]

        panel_w = 200
        panel_h = 175
        panel_x = w - panel_w - 10   # right edge
        panel_y = 10                  # top edge — clear of everything

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Border
        cv2.rectangle(frame,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h),
                      (0, 255, 255), 1)

        # Title
        cv2.putText(frame, "VEHICLE COUNTS",
                    (panel_x + 8, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 255, 255), 1)
        cv2.line(frame,
                 (panel_x + 4, panel_y + 27),
                 (panel_x + panel_w - 4, panel_y + 27),
                 (0, 255, 255), 1)

        colors = {
            'Car':        (0, 255, 0),
            'Motorcycle': (0, 165, 255),
            'Bus':        (255, 0, 255),
            'Truck':      (255, 165, 0),
        }
        icons = {
            'Car': 'CAR', 'Motorcycle': 'MOTO',
            'Bus': 'BUS', 'Truck': 'TRUCK',
        }

        y = panel_y + 48
        for label, count in self.counts.items():
            c = colors.get(label, (255, 255, 255))

            # Small colour swatch
            cv2.rectangle(frame,
                          (panel_x + 8, y - 11),
                          (panel_x + 18, y + 1),
                          c, -1)

            cv2.putText(frame, f"{icons[label]}: {count}",
                        (panel_x + 24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, c, 1)
            y += 26

        # Divider + total
        cv2.line(frame,
                 (panel_x + 4, y + 2),
                 (panel_x + panel_w - 4, y + 2),
                 (0, 255, 255), 1)
        cv2.putText(frame, f"TOTAL: {self.total}",
                    (panel_x + 8, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 255), 2)
import os
import time
import json
import math
import random
import tempfile
import threading
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import streamlit as st

try:
    import av
except Exception:
    av = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
except Exception:
    webrtc_streamer = None
    WebRtcMode = None

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:
    streamlit_js_eval = None

import streamlit.components.v1 as components


st.set_page_config(
    page_title="AirDrum Punch Bridge",
    page_icon="🥁",
    layout="wide",
)

QUADRANT_KEYS = ["D", "F", "J", "K"]
QUADRANT_LABELS = ["Top-left", "Top-right", "Bottom-left", "Bottom-right"]
DEFAULT_THRESHOLD_PX_S = 1300.0
DEFAULT_COOLDOWN_MS = 135
CALIBRATION_SECONDS = 10.0


def auto_refresh_fragment(run_every):
    """Compatibility wrapper for Streamlit fragments."""
    if hasattr(st, "fragment"):
        return st.fragment(run_every=run_every)
    if hasattr(st, "experimental_fragment"):
        return st.experimental_fragment(run_every=run_every)
    return lambda func: func


def quadrant_rect(q, w, h):
    if q == 0:
        return 0, 0, w // 2, h // 2
    if q == 1:
        return w // 2, 0, w - w // 2, h // 2
    if q == 2:
        return 0, h // 2, w // 2, h - h // 2
    return w // 2, h // 2, w - w // 2, h - h // 2


def quadrant_center(q, w, h):
    x, y, ww, hh = quadrant_rect(q, w, h)
    return int(x + ww / 2), int(y + hh / 2)


def quadrant_from_point(x, y, w, h):
    if x < w / 2 and y < h / 2:
        return 0
    if x >= w / 2 and y < h / 2:
        return 1
    if x < w / 2 and y >= h / 2:
        return 2
    return 3


def draw_badge(img, text, origin=(10, 10), bg=(0, 0, 0), fg=(255, 255, 255), scale=0.55, thickness=1):
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (x - 4, y - 4), (x + tw + 8, y + th + baseline + 8), bg, -1)
    cv2.putText(img, text, (x, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def draw_grid(img, flash_times=None, now=None):
    h, w = img.shape[:2]

    if flash_times:
        layer = np.zeros_like(img)
        any_flash = False
        for q, t in flash_times.items():
            if now is None or now - t < 0.35:
                alpha = 1.0 if now is None else max(0.0, 1.0 - (now - t) / 0.35)
                if alpha <= 0:
                    continue
                x, y, ww, hh = quadrant_rect(q, w, h)
                color = (0, 255, 170) if q in (0, 3) else (80, 200, 255)
                color = tuple(int(c * alpha) for c in color)
                cv2.rectangle(layer, (x, y), (x + ww, y + hh), color, -1)
                any_flash = True
        if any_flash:
            img = cv2.addWeighted(layer, 0.42, img, 1.0, 0.0)

    cv2.line(img, (w // 2, 0), (w // 2, h), (0, 220, 255), 2)
    cv2.line(img, (0, h // 2), (w, h // 2), (0, 220, 255), 2)

    for q, key in enumerate(QUADRANT_KEYS):
        x, y, _, _ = quadrant_rect(q, w, h)
        label = f"{key} | {QUADRANT_LABELS[q]}"
        cv2.putText(img, label, (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def draw_fake_hand(img, center, label, velocity, threshold):
    cx, cy = center
    color = (0, 200, 255) if label == "Left" else (255, 120, 0)
    offsets = [(-42, -58), (-16, -80), (10, -82), (36, -64), (58, -30)]

    for dx, dy in offsets:
        pt = (int(cx + dx), int(cy + dy))
        cv2.line(img, center, pt, color, 2)
        cv2.circle(img, pt, 4, color, -1)

    cv2.circle(img, center, 13, color, -1)
    cv2.circle(img, center, 22, (255, 255, 255), 1)

    status_color = (0, 255, 0) if velocity >= threshold else (0, 255, 255)
    text = f"{label} {int(velocity)} px/s"
    cv2.putText(img, text, (cx - 55, cy + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)


class HitLogger:
    def __init__(self, maxlen=5000):
        self.lock = threading.RLock()
        self.events = collections.deque(maxlen=maxlen)
        self.new_events = collections.deque(maxlen=maxlen)
        self.latencies = collections.deque(maxlen=maxlen)
        self.accuracies = collections.deque(maxlen=maxlen)

    def add_event(self, event):
        with self.lock:
            self.events.append(event)
            self.new_events.append(event)
            if event.get("beat_error_ms") is not None:
                self.accuracies.append(float(event["beat_error_ms"]))

    def add_latency(self, ms):
        with self.lock:
            self.latencies.append(float(ms))

    def pop_new(self):
        with self.lock:
            items = list(self.new_events)
            self.new_events.clear()
            return items

    def df(self):
        with self.lock:
            if not self.events:
                return pd.DataFrame(
                    columns=[
                        "id",
                        "time",
                        "quadrant",
                        "lane",
                        "hand",
                        "velocity_px_s",
                        "threshold_px_s",
                        "source",
                        "beat_error_ms",
                    ]
                )
            return pd.DataFrame(list(self.events))

    def clear(self):
        with self.lock:
            self.events.clear()
            self.new_events.clear()
            self.latencies.clear()
            self.accuracies.clear()


class PunchPipeline:
    def __init__(self):
        self.lock = threading.RLock()
        self.logger = HitLogger()

        self.start_time = time.time()
        self.bpm = 112.0
        self.cooldown_ms = DEFAULT_COOLDOWN_MS
        self.threshold_px_s = DEFAULT_THRESHOLD_PX_S
        self.auto_threshold = True

        self.prev = {}
        self.vel = {}
        self.cooldown_until = {}
        self.flash = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        self.last_gesture = None
        self.event_id = 0

        self.calibrating = False
        self.calib_start = None
        self.calib_speeds = []
        self.calib_positions = []
        self.calib_message = ""

        self.hands = None
        self.mp_hands = None
        self.mp_draw = None
        self.hands_failed = False

        self.recording = False
        self.recording_requested = False
        self.video_writer = None
        self.video_path = None
        self.record_start = None

    def configure(self, bpm, cooldown_ms, manual_threshold, auto_threshold):
        with self.lock:
            self.bpm = float(bpm)
            self.cooldown_ms = float(cooldown_ms)
            self.auto_threshold = bool(auto_threshold)
            if not auto_threshold:
                self.threshold_px_s = float(manual_threshold)

    def ensure_hands(self):
        if mp is None or self.hands is not None or self.hands_failed:
            return
        try:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            self.hands = None
            self.hands_failed = True

    def start_calibration(self):
        with self.lock:
            self.calibrating = True
            self.calib_start = time.perf_counter()
            self.calib_speeds = []
            self.calib_positions = []
            self.calib_message = "Calibration started: punch naturally for 10 seconds."

    def finish_calibration(self):
        speeds = np.array(self.calib_speeds, dtype=float) if self.calib_speeds else np.array([])
        if len(speeds) > 8:
            p70 = float(np.percentile(speeds, 70))
            p95 = float(np.percentile(speeds, 95))
            candidate = 0.55 * p95 + 0.15 * p70
            self.threshold_px_s = float(np.clip(candidate, 700.0, 2800.0))
            self.auto_threshold = True
            self.calib_message = f"Calibration complete: threshold set to {int(self.threshold_px_s)} px/s."
        else:
            self.calib_message = "Calibration complete: kept previous threshold."
        self.calibrating = False
        self.calib_speeds = []
        self.calib_positions = []

    def arm_recording(self):
        self.recording_requested = True

    def maybe_start_recording(self, w, h, fps=24):
        if self.recording_requested and not self.recording:
            self.start_recording(w, h, fps)
            self.recording_requested = False

    def start_recording(self, w, h, fps=24):
        if self.recording:
            return self.video_path

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(tempfile.gettempdir(), f"airdrum_session_{stamp}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(w), int(h)))

        if not writer.isOpened():
            path = os.path.join(tempfile.gettempdir(), f"airdrum_session_{stamp}.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (int(w), int(h)))

        if writer.isOpened():
            self.video_writer = writer
            self.recording = True
            self.video_path = path
            self.record_start = time.time()

        return self.video_path

    def stop_recording(self):
        if not self.recording:
            return self.video_path

        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        return self.video_path

    def record_frame(self, img):
        if img is None:
            return
        h, w = img.shape[:2]
        self.maybe_start_recording(w, h)
        if self.recording and self.video_writer is not None:
            self.video_writer.write(img)

    def _beat_error_ms(self, wall_time):
        if self.bpm <= 0:
            return None
        t = wall_time - self.start_time
        interval = 60.0 / self.bpm
        phase = t % interval
        return float(min(phase, interval - phase) * 1000.0)

    def register_event(self, quadrant, hand, velocity, source, detect_latency_ms=None, ignore_cooldown=False):
        if quadrant not in (0, 1, 2, 3):
            return None

        now = time.perf_counter()
        wall = time.time()

        with self.lock:
            if not ignore_cooldown and now < self.cooldown_until.get(quadrant, 0.0):
                return None

            self.cooldown_until[quadrant] = now + self.cooldown_ms / 1000.0
            self.flash[quadrant] = now
            self.event_id += 1

            lane = QUADRANT_KEYS[quadrant]
            beat_error = self._beat_error_ms(wall)

            event = {
                "id": self.event_id,
                "time": datetime.fromtimestamp(wall).strftime("%H:%M:%S.%f")[:-3],
                "wall_time": wall,
                "quadrant": quadrant,
                "lane": lane,
                "hand": hand,
                "velocity_px_s": float(velocity),
                "threshold_px_s": float(self.threshold_px_s),
                "source": source,
                "detect_latency_ms": detect_latency_ms,
                "beat_error_ms": beat_error,
            }

            self.logger.add_event(event)
            self.last_gesture = f"{hand} {QUADRANT_LABELS[quadrant]} -> {lane}"
            return event

    def reset_state(self):
        with self.lock:
            self.prev.clear()
            self.vel.clear()
            self.cooldown_until.clear()
            self.flash = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
            self.last_gesture = None
            self.calibrating = False
            self.calib_speeds = []
            self.calib_positions = []
            self.calib_message = ""
            self.logger.clear()

    def process_bgr(self, img_bgr, frame_time=None):
        h, w = img_bgr.shape[:2]
        arrival = frame_time if frame_time is not None else time.perf_counter()
        annotated = img_bgr.copy()
        now = time.perf_counter()

        self.ensure_hands()
        hand_infos = []

        if self.hands is not None:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                handedness_list = results.multi_handedness or []

                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    handedness = handedness_list[idx] if idx < len(handedness_list) else None

                    if handedness is not None and handedness.classification:
                        label = handedness.classification[0].label
                    else:
                        label = f"Hand {idx}"

                    lm = hand_lms.landmark
                    palm_ids = [0, 5, 9, 13, 17]
                    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in palm_ids], dtype=float)
                    center = pts.mean(axis=0)

                    if self.mp_draw is not None and self.mp_hands is not None:
                        self.mp_draw.draw_landmarks(annotated, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                    cv2.circle(annotated, (int(center[0]), int(center[1])), 8, (0, 255, 0), -1)

                    prev = self.prev.get(label)
                    vel = 0.0

                    if prev is not None:
                        dt = now - prev[0]
                        if 0.003 < dt < 0.5:
                            vel = float((center[1] - prev[1]) / dt)
                        else:
                            vel = 0.0

                    down_vel = max(0.0, vel)
                    ema_vel = 0.55 * down_vel + 0.45 * self.vel.get(label, down_vel)
                    self.vel[label] = ema_vel
                    self.prev[label] = (now, float(center[0]), float(center[1]))

                    q = quadrant_from_point(center[0], center[1], w, h)
                    hand_infos.append((label, center, ema_vel, q))

                    if not self.calibrating and ema_vel > self.threshold_px_s:
                        self.register_event(
                            q,
                            label,
                            ema_vel,
                            "live",
                            detect_latency_ms=(now - arrival) * 1000.0,
                        )

                    cv2.putText(
                        annotated,
                        f"{label}: {int(ema_vel)} px/s",
                        (int(center[0]) - 45, int(center[1]) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0) if ema_vel >= self.threshold_px_s else (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        if self.calibrating:
            elapsed = now - self.calib_start
            for _, center, vel, _ in hand_infos:
                self.calib_positions.append((float(center[0]), float(center[1])))
                if vel > 0:
                    self.calib_speeds.append(float(vel))

            if elapsed >= CALIBRATION_SECONDS:
                self.finish_calibration()

        annotated = draw_grid(annotated, self.flash, now)

        if mp is None:
            draw_badge(annotated, "MediaPipe not installed. Live hand tracking disabled.", (10, 106), bg=(0, 0, 120))

        if self.calibrating:
            elapsed = now - self.calib_start
            remaining = max(0.0, CALIBRATION_SECONDS - elapsed)
            cv2.rectangle(annotated, (w // 2 - 130, h // 2 - 90), (w // 2 + 130, h // 2 + 90), (0, 255, 255), 2)
            draw_badge(annotated, f"CALIBRATION: punch naturally for {remaining:.1f}s", (10, h - 40))

        draw_badge(annotated, f"Last gesture: {self.last_gesture or 'Waiting for punch'}", (10, 10))
        draw_badge(
            annotated,
            f"Threshold: {int(self.threshold_px_s)} px/s | Cooldown: {int(self.cooldown_ms)} ms | BPM: {int(self.bpm)}",
            (10, 42),
        )

        for i, (label, _, vel, q) in enumerate(hand_infos):
            draw_badge(annotated, f"{label} velocity: {int(vel)} px/s | zone: {QUADRANT_LABELS[q]}", (10, 74 + i * 30))

        if self.calib_message:
            draw_badge(annotated, self.calib_message, (10, h - 70), bg=(0, 80, 0))

        self.record_frame(annotated)
        return annotated


class DemoSimulator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.w, self.h = 640, 480
        self.start = time.perf_counter()
        self.last = self.start

        self.positions = {
            "Left": np.array([self.w * 0.34, self.h * 0.52], dtype=float),
            "Right": np.array([self.w * 0.66, self.h * 0.52], dtype=float),
        }
        self.targets = {k: v.copy() for k, v in self.positions.items()}
        self.display_vel = {"Left": 0.0, "Right": 0.0}

        self.next_event_cycle = 16.0
        self.slow_seq = [0, 1, 2, 3, 0, 2, 1, 3]
        self.slow_idx = 0
        self.last_phase = None

    def step(self):
        now = time.perf_counter()
        dt = max(0.008, min(0.08, now - self.last))
        self.last = now

        t = now - self.start
        cycle = t % 120.0

        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (22, 20, 34)

        if cycle < 15:
            phase = "Calibration warm-up"
        elif cycle < 45:
            phase = "Slow deliberate punches"
        elif cycle < 90:
            phase = "Fast gameplay"
        else:
            phase = "Near-miss / edge cases"

        if phase != self.last_phase:
            self.last_phase = phase
            if phase == "Slow deliberate punches":
                self.next_event_cycle = cycle + 1.2
            elif phase == "Fast gameplay":
                self.next_event_cycle = cycle + 0.7
            elif phase == "Near-miss / edge cases":
                self.next_event_cycle = cycle + 1.0

        if phase == "Calibration warm-up":
            box_center = np.array([self.w / 2, self.h / 2], dtype=float)
            self.targets["Left"] = box_center + np.array([-70, 10 * math.sin(now * 1.7)])
            self.targets["Right"] = box_center + np.array([70, 10 * math.cos(now * 1.9)])
            self.display_vel["Left"] = max(0.0, self.display_vel["Left"] * 0.9)
            self.display_vel["Right"] = max(0.0, self.display_vel["Right"] * 0.9)

        elif cycle >= self.next_event_cycle:
            if phase == "Slow deliberate punches":
                q = self.slow_seq[self.slow_idx % len(self.slow_seq)]
                self.slow_idx += 1
                hand = "Left" if q in (0, 2) else "Right"
                velocity = max(self.pipeline.threshold_px_s * random.uniform(1.18, 1.45), 1200.0)
                interval = random.uniform(2.2, 2.8)

                self.targets[hand] = np.array(quadrant_center(q, self.w, self.h), dtype=float)
                self.display_vel[hand] = velocity
                self.pipeline.register_event(q, hand, velocity, "demo", detect_latency_ms=random.uniform(4, 10))

            elif phase == "Fast gameplay":
                q = random.randint(0, 3)
                hand = random.choice(["Left", "Right"])
                velocity = max(self.pipeline.threshold_px_s * random.uniform(1.25, 2.1), 1450.0)
                interval = random.uniform(0.62, 1.08)

                self.targets[hand] = np.array(quadrant_center(q, self.w, self.h), dtype=float)
                self.display_vel[hand] = velocity
                self.pipeline.register_event(q, hand, velocity, "demo", detect_latency_ms=random.uniform(3, 9))

            else:
                hand = random.choice(["Left", "Right"])

                if random.random() < 0.22:
                    q = random.randint(0, 3)
                    velocity = max(self.pipeline.threshold_px_s * random.uniform(1.15, 1.35), 1250.0)
                    self.targets[hand] = np.array(quadrant_center(q, self.w, self.h), dtype=float)
                    self.display_vel[hand] = velocity
                    self.pipeline.register_event(q, hand, velocity, "demo-edge", detect_latency_ms=random.uniform(4, 11))
                    interval = random.uniform(1.8, 2.6)
                else:
                    velocity = self.pipeline.threshold_px_s * random.uniform(0.22, 0.58)
                    jitter = np.array([random.uniform(-45, 45), random.uniform(-28, 34)], dtype=float)
                    self.targets[hand] = self.positions[hand] + jitter
                    self.display_vel[hand] = velocity
                    interval = random.uniform(0.75, 1.35)

            self.next_event_cycle = cycle + interval

        for hand in self.positions:
            prev_pos = self.positions[hand].copy()
            alpha = min(1.0, dt * 7.5)
            self.positions[hand] += (self.targets[hand] - self.positions[hand]) * alpha

            if dt > 0:
                actual_v = float(np.linalg.norm(self.positions[hand] - prev_pos) / dt)
                self.display_vel[hand] = max(actual_v, self.display_vel[hand] * 0.88)

        img = draw_grid(img, self.pipeline.flash, now)

        if phase == "Calibration warm-up":
            cv2.rectangle(img, (self.w // 2 - 140, self.h // 2 - 100), (self.w // 2 + 140, self.h // 2 + 100), (0, 255, 255), 2)
            draw_badge(img, "Hold hand steady in the box, then punch naturally during calibration", (10, self.h - 40))

        for hand, pos in self.positions.items():
            draw_fake_hand(img, (int(pos[0]), int(pos[1])), hand, self.display_vel[hand], self.pipeline.threshold_px_s)

        draw_badge(img, f"Last gesture: {self.pipeline.last_gesture or 'Waiting for punch'}", (10, 10))
        draw_badge(
            img,
            f"Threshold {int(self.pipeline.threshold_px_s)} px/s | Cooldown {int(self.pipeline.cooldown_ms)} ms | BPM {int(self.pipeline.bpm)}",
            (10, 42),
        )
        draw_badge(img, f"Demo phase: {phase} | {cycle:05.1f}s / 120s", (10, 74))

        if self.pipeline.calibrating:
            draw_badge(img, "Live calibration armed (use webcam mode for real hand tracking)", (10, 106))

        self.pipeline.record_frame(img)
        return img


GAME_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {
  margin: 0;
  background: #0b1020;
  color: #eaf7ff;
  font-family: Arial, sans-serif;
}
.wrap {
  padding: 10px;
}
h3 {
  margin: 0 0 8px 0;
  font-size: 15px;
}
#status {
  font-size: 12px;
  margin-top: 8px;
  color: #9ee8ff;
}
canvas {
  background: #070b16;
  border: 1px solid #29496b;
  border-radius: 8px;
}
</style>
</head>
<body>
<div class="wrap">
<h3>Lane rhythm game (D/F/J/K)</h3>
<canvas id="game" width="380" height="360"></canvas>
<div id="status">Waiting for gesture hits...</div>
</div>
<script>
const canvas = document.getElementById('game');
const statusEl = document.getElementById('status');
const ctx = canvas.getContext('2d');

let bpm = __BPM__;
let beatMs = 60000 / bpm;
let lastBeat = performance.now();

const hitY = canvas.height - 58;
const fallTime = 1800;

const lanes = [
  {key: 'D', color: '#ff5d73', flash: 0, lastHit: 'idle'},
  {key: 'F', color: '#ffd166', flash: 0, lastHit: 'idle'},
  {key: 'J', color: '#06d6a0', flash: 0, lastHit: 'idle'},
  {key: 'K', color: '#4cc9f0', flash: 0, lastHit: 'idle'}
];

let notes = [];
let score = 0;
let combo = 0;
let best = 0;

function spawnNote() {
  const lane = Math.floor(Math.random() * 4);
  notes.push({lane: lane, born: performance.now(), hit: false});
}

function hitLane(laneIdx, key, source) {
  const now = performance.now();
  lanes[laneIdx].flash = now;
  lanes[laneIdx].lastHit = source || 'gesture';

  let bestNote = null;
  let bestDist = 1e9;

  for (const n of notes) {
    if (n.lane != laneIdx || n.hit) continue;
    const y = ((now - n.born) / fallTime) * canvas.height;
    const dist = Math.abs(y - hitY);
    if (dist < bestDist) {
      bestDist = dist;
      bestNote = n;
    }
  }

  let judge = 'Miss';
  if (bestNote && bestDist < 24) {
    judge = 'Perfect';
    score += 300;
    combo++;
    bestNote.hit = true;
  } else if (bestNote && bestDist < 62) {
    judge = 'Great';
    score += 150;
    combo++;
    bestNote.hit = true;
  } else {
    combo = 0;
  }

  best = Math.max(best, combo);
  statusEl.textContent = 'Lane ' + key + ' ' + judge + ' | score ' + score + ' | combo ' + combo;

  try {
    const opts = {
      key: key.toLowerCase(),
      code: 'Key' + key.toUpperCase(),
      bubbles: true,
      cancelable: true
    };
    document.dispatchEvent(new KeyboardEvent('keydown', opts));
    document.dispatchEvent(new KeyboardEvent('keyup', opts));
  } catch (e) {}
}

window.addEventListener('message', (e) => {
  if (!e.data) return;

  if (e.data.type === 'drum-hit') {
    hitLane(e.data.lane || 0, e.data.key || 'D', e.data.source || 'external');
  }

  if (e.data.type === 'drum-hit-batch' && Array.isArray(e.data.hits)) {
    e.data.hits.forEach(h => hitLane(h.lane || 0, h.key || 'D', h.source || 'gesture'));
  }
});

document.addEventListener('keydown', (e) => {
  const idx = ['d', 'f', 'j', 'k'].indexOf(String(e.key || '').toLowerCase());
  if (idx >= 0) {
    hitLane(idx, ['D', 'F', 'J', 'K'][idx], 'keyboard');
  }
});

function draw() {
  const now = performance.now();

  if (now - lastBeat > beatMs) {
    spawnNote();
    if (Math.random() < 0.2) spawnNote();
    lastBeat = now;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const laneW = canvas.width / 4;

  lanes.forEach((lane, i) => {
    const x = i * laneW;
    ctx.fillStyle = 'rgba(255,255,255,0.045)';
    ctx.fillRect(x + 2, 0, laneW - 4, canvas.height);

    const flashAge = now - lane.flash;
    if (flashAge < 220) {
      const alpha = Math.floor((1 - flashAge / 220) * 90).toString(16).padStart(2, '0');
      ctx.fillStyle = lane.color + alpha;
      ctx.fillRect(x + 2, 0, laneW - 4, canvas.height);
    }

    ctx.fillStyle = lane.color;
    ctx.font = 'bold 20px Arial';
    ctx.fillText(lane.key, x + laneW / 2 - 7, canvas.height - 16);
  });

  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  ctx.beginPath();
  ctx.moveTo(0, hitY);
  ctx.lineTo(canvas.width, hitY);
  ctx.stroke();

  notes = notes.filter(n => {
    const y = ((now - n.born) / fallTime) * canvas.height;

    if (n.hit && y > hitY + 35) return false;

    if (y > canvas.height + 30) {
      if (!n.hit) combo = 0;
      return false;
    }

    const x = n.lane * laneW + laneW / 2;
    ctx.beginPath();
    ctx.fillStyle = lanes[n.lane].color;
    ctx.arc(x, y, 12, 0, Math.PI * 2);
    ctx.fill();

    return true;
  });

  ctx.fillStyle = 'white';
  ctx.font = '12px Arial';
  ctx.fillText('Score ' + score + '  Combo ' + combo + '  BPM ' + bpm, 8, 16);

  requestAnimationFrame(draw);
}

draw();
</script>
</body>
</html>
"""


def render_game(bpm):
    html = GAME_HTML_TEMPLATE.replace("__BPM__", str(int(bpm)))
    components.html(html, height=470, scrolling=False)


def build_dispatch_js(hits):
    payload = json.dumps({"type": "drum-hit-batch", "hits": hits})
    return f"""
    (function() {{
      const payload = {payload};

      try {{
        document.querySelectorAll('iframe').forEach(function(frame) {{
          try {{
            if (frame.contentWindow) {{
              frame.contentWindow.postMessage(payload, '*');
            }}
          }} catch (e) {{}}
        }});
      }} catch (e) {{}}

      try {{
        payload.hits.forEach(function(h) {{
          const key = (h.key || 'd').toLowerCase();
          const opts = {{key: key, bubbles: true, cancelable: true}};
          document.dispatchEvent(new KeyboardEvent('keydown', opts));
          document.dispatchEvent(new KeyboardEvent('keyup', opts));
        }});
      }} catch (e) {{}}
    }})();
    """


@st.cache_resource
def get_pipeline():
    return PunchPipeline()


@st.cache_resource
def get_demo_sim():
    return DemoSimulator(get_pipeline())


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    t0 = time.perf_counter()
    pipeline = get_pipeline()
    out = pipeline.process_bgr(img, frame_time=t0)

    if av is None:
        return frame

    return av.VideoFrame.from_ndarray(out, format="bgr24")


def live_view():
    if webrtc_streamer is None or WebRtcMode is None:
        st.error("streamlit-webrtc is not installed. Install requirements and restart.")
        return

    if mp is None:
        st.warning("MediaPipe is not installed. Live hand tracking will be disabled.")

    st.caption("Allow camera access if prompted. The overlay shows quadrants, hand landmarks, hit flashes, and velocity.")

    webrtc_streamer(
        key="airdrum-live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,
    )


@auto_refresh_fragment(run_every=0.04)
def demo_view():
    sim = get_demo_sim()
    pipeline = get_pipeline()

    img = sim.step()

    if (
        pipeline.recording
        and st.session_state.get("auto_record_demo", False)
        and pipeline.record_start is not None
        and time.time() - pipeline.record_start > 120.0
        and not st.session_state.get("demo_record_done", False)
    ):
        path = pipeline.stop_recording()
        st.session_state.video_path = path
        st.session_state.demo_record_done = True
        try:
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.session_state.video_bytes = f.read()
        except Exception:
            st.session_state.video_bytes = None

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)


@auto_refresh_fragment(run_every=0.10)
def live_bridge():
    pipeline = get_pipeline()
    new_events = pipeline.logger.pop_new()

    if new_events:
        hits = []
        for e in new_events:
            dispatch_delay = max(0.0, (time.time() - e.get("wall_time", time.time())) * 1000.0)
            pipeline.logger.add_latency(dispatch_delay)

            hits.append(
                {
                    "lane": int(e["quadrant"]),
                    "key": e["lane"],
                    "hand": e.get("hand", "?"),
                    "source": e.get("source", "gesture"),
                    "velocity": int(e.get("velocity_px_s", 0)),
                }
            )

        if streamlit_js_eval is not None and hits:
            streamlit_js_eval(build_dispatch_js(hits))

    events_df = pipeline.logger.df()
    latencies = list(pipeline.logger.latencies)
    accuracies = list(pipeline.logger.accuracies)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total hits", len(events_df))

    if latencies:
        c2.metric("Last dispatch latency", f"{latencies[-1]:.0f} ms", f"median {np.median(latencies):.0f} ms")
    else:
        c2.metric("Last dispatch latency", "--")

    if accuracies:
        c3.metric("Last beat error", f"{accuracies[-1]:.0f} ms", f"mean50 {np.mean(accuracies[-50:]):.0f} ms")
    else:
        c3.metric("Last beat error", "--")

    if streamlit_js_eval is None:
        st.warning("streamlit-js-eval is not active. Hits will be logged, but JS dispatch may not work.")

    if len(events_df):
        display_df = events_df.drop(columns=["wall_time"], errors="ignore")
        st.dataframe(display_df.sort_values("id", ascending=False).head(8), use_container_width=True)
    else:
        st.info("No hits yet. Punch a quadrant or use the manual pads.")


@auto_refresh_fragment(run_every=0.50)
def latency_histogram():
    pipeline = get_pipeline()
    lat = list(pipeline.logger.latencies)

    st.markdown("### Hit latency histogram")

    if not lat:
        st.info("No latency samples yet.")
        return

    s = pd.Series(lat, dtype=float)
    upper = max(60.0, float(s.quantile(0.98)))
    bins = np.linspace(0, upper, 26)
    hist = pd.cut(s.clip(0, upper), bins=bins).value_counts().sort_index()

    chart_df = pd.DataFrame(
        {
            "Latency bucket": hist.index.astype(str),
            "Hits": hist.values,
        }
    )

    st.bar_chart(chart_df.set_index("Latency bucket"))
    st.caption(
        f"p50 {s.quantile(0.5):.0f} ms • p95 {s.quantile(0.95):.0f} ms • max {s.max():.0f} ms"
    )


@auto_refresh_fragment(run_every=1.0)
def downloads_panel():
    pipeline = get_pipeline()

    df = pipeline.logger.df()
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV event log",
        data=csv,
        file_name="airdrum_events.csv",
        mime="text/csv",
    )

    video_path = st.session_state.get("video_path") or pipeline.video_path
    video_bytes = st.session_state.get("video_bytes")

    if video_bytes and video_path and not pipeline.recording:
        mime = "video/mp4" if video_path.endswith(".mp4") else "video/avi"
        st.download_button(
            "Download session video",
            data=video_bytes,
            file_name=os.path.basename(video_path),
            mime=mime,
        )
    else:
        st.caption("Stop recording to enable video download.")


def manual_pads():
    st.write("Manual test pads")
    cols = st.columns(4)
    pipeline = get_pipeline()

    for i, key in enumerate(QUADRANT_KEYS):
        if cols[i].button(key, key=f"manual_pad_{i}"):
            pipeline.register_event(
                i,
                "Manual",
                pipeline.threshold_px_s * 1.2,
                "manual",
                detect_latency_ms=0.0,
                ignore_cooldown=True,
            )


st.title("🥁 AirDrum: webcam punch-to-key rhythm bridge")
st.caption(
    "Four quadrant drum pads + MediaPipe hand velocity detection + low-latency browser JS key dispatch. "
    "Instant demo starts automatically. Switch to live webcam mode for real-camera tracking."
)

pipeline = get_pipeline()

st.sidebar.title("Controls")

mode = st.sidebar.radio(
    "Pipeline source",
    ["Instant demo (auto-start)", "Live webcam (real camera)"],
    index=0,
    help="Instant demo prevents a blank dashboard. Live webcam uses streamlit-webrtc and MediaPipe Hands.",
)

bpm = st.sidebar.slider("Song BPM (beat target)", 60, 180, 112)
cooldown_ms = st.sidebar.slider("Quadrant cooldown (ms)", 80, 250, int(DEFAULT_COOLDOWN_MS))
auto_threshold = st.sidebar.checkbox("Auto velocity threshold", value=True)
manual_threshold = st.sidebar.slider(
    "Manual threshold (px/s)",
    500,
    3000,
    int(DEFAULT_THRESHOLD_PX_S),
    disabled=auto_threshold,
)

st.sidebar.caption("Live calibration measures punch speed for 10 seconds and auto-tunes the threshold.")

if st.sidebar.button("Start 10s calibration"):
    if mode.startswith("Live"):
        pipeline.start_calibration()
    else:
        pipeline.threshold_px_s = float(np.random.uniform(1150, 1450))
        st.sidebar.success("Demo calibration simulated.")

if st.sidebar.button("Reset session"):
    pipeline.reset_state()
    get_demo_sim.clear()
    st.session_state.video_path = None
    st.session_state.video_bytes = None
    st.session_state.demo_record_done = False
    st.rerun()

st.sidebar.markdown("---")

record_label = "Stop recording" if pipeline.recording else "Start recording"

if st.sidebar.button(record_label):
    if pipeline.recording:
        path = pipeline.stop_recording()
        st.session_state.video_path = path
        st.session_state.demo_record_done = True
        try:
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    st.session_state.video_bytes = f.read()
        except Exception:
            st.session_state.video_bytes = None
    else:
        st.session_state.video_bytes = None
        st.session_state.video_path = None

        if mode.startswith("Instant"):
            pipeline.start_recording(640, 480, fps=24)
        else:
            pipeline.arm_recording()

auto_record_demo = st.sidebar.checkbox("Auto-record first 2-minute demo", value=True)

st.sidebar.markdown("---")

with st.sidebar:
    downloads_panel()

st.sidebar.markdown("---")

with st.sidebar:
    latency_histogram()

st.session_state.bpm = bpm
st.session_state.auto_record_demo = auto_record_demo

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None

if "demo_record_done" not in st.session_state:
    st.session_state.demo_record_done = False

pipeline.configure(
    bpm=bpm,
    cooldown_ms=cooldown_ms,
    manual_threshold=manual_threshold,
    auto_threshold=auto_threshold,
)

if (
    mode.startswith("Instant")
    and auto_record_demo
    and not pipeline.recording
    and not st.session_state.demo_record_done
):
    pipeline.start_recording(640, 480, fps=24)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Camera / annotated view")

    if mode.startswith("Instant"):
        demo_view()
    else:
        live_view()

with right:
    st.subheader("Embedded rhythm game")
    render_game(bpm)

    st.caption("The bridge posts gesture hits to the embedded iframe and also dispatches D/F/J/K keyboard events.")

    manual_pads()
    live_bridge()
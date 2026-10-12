#!/usr/bin/env python3

import json
import random
import sys
from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QPoint, QParallelAnimationGroup, QPropertyAnimation, QRect, Qt, QTimer, QUrl, pyqtProperty
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPalette,
    QPen,
    QRadialGradient,
    QImage,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import io
import math
import struct
import tempfile
import os

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


class SoundEngine:
    """Generates and plays procedural sound effects via QSoundEffect.

    All WAV files are written once to a temp directory at startup and
    reused for every play call, so there is no per-draw file I/O.
    """

    RATE = 22050

    def __init__(self):
        self._tmpdir = tempfile.mkdtemp(prefix="keno_sfx_")
        self._effects: dict[str, QSoundEffect] = {}
        if _NUMPY:
            self._build_all()

    # ── WAV generation helpers ─────────────────────────────────────────────

    @staticmethod
    def _to_wav_bytes(samples: "np.ndarray", rate: int) -> bytes:
        """Convert a float32 numpy array (−1…1) to 16-bit mono WAV bytes."""
        pcm = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm * 32767).astype(np.int16)
        buf = io.BytesIO()
        import wave as _wave
        with _wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def _save(self, name: str, samples: "np.ndarray"):
        path = os.path.join(self._tmpdir, f"{name}.wav")
        data = self._to_wav_bytes(samples, self.RATE)
        with open(path, "wb") as f:
            f.write(data)
        fx = QSoundEffect()
        fx.setSource(QUrl.fromLocalFile(path))
        fx.setVolume(0.55)
        self._effects[name] = fx

    # ── Sound definitions ──────────────────────────────────────────────────

    def _build_all(self):
        r = self.RATE
        t_short = np.linspace(0, 0.06, int(r * 0.06), endpoint=False)
        t_med   = np.linspace(0, 0.12, int(r * 0.12), endpoint=False)
        t_long  = np.linspace(0, 0.55, int(r * 0.55), endpoint=False)

        # ── tick: soft high-pitched click for a normal (miss) ball drop
        env_tick = np.exp(-t_short * 80)
        tick = np.sin(2 * np.pi * 880 * t_short) * env_tick * 0.4
        tick += np.sin(2 * np.pi * 1320 * t_short) * env_tick * 0.2
        self._save("tick", tick)

        # ── hit: bright chime for a matching ball drop
        env_hit = np.exp(-t_med * 35)
        hit = (
            np.sin(2 * np.pi * 1047 * t_med) * 0.5
            + np.sin(2 * np.pi * 1319 * t_med) * 0.3
            + np.sin(2 * np.pi * 1568 * t_med) * 0.2
        ) * env_hit
        self._save("hit", hit)

        # ── coin: short metallic clink for each winnings counter tick
        env_coin = np.exp(-t_short * 60)
        coin = (
            np.sin(2 * np.pi * 1200 * t_short) * 0.45
            + np.sin(2 * np.pi * 2400 * t_short) * 0.2
            + np.random.default_rng(0).uniform(-0.08, 0.08, len(t_short))
        ) * env_coin
        self._save("coin", coin)

        # ── win_fanfare: ascending arpeggio when winnings count finishes
        notes = [523, 659, 784, 1047]  # C5 E5 G5 C6
        fanfare = np.zeros(int(r * 0.55))
        step = int(r * 0.12)
        for i, freq in enumerate(notes):
            start = i * int(r * 0.10)
            end   = min(start + step, len(fanfare))
            seg_t = np.linspace(0, (end - start) / r, end - start, endpoint=False)
            env   = np.exp(-seg_t * 18)
            fanfare[start:end] += (
                np.sin(2 * np.pi * freq * seg_t) * 0.45
                + np.sin(2 * np.pi * freq * 2 * seg_t) * 0.15
            ) * env
        fanfare /= np.max(np.abs(fanfare) + 1e-9)
        fanfare *= 0.75
        self._save("win_fanfare", fanfare)

        # ── jackpot_fanfare: big triumphant burst for a progressive jackpot win
        jt = np.linspace(0, 1.2, int(r * 1.2), endpoint=False)
        chord_freqs = [523, 659, 784, 1047, 1319]  # C E G C E
        jfanfare = np.zeros(len(jt))
        for i, freq in enumerate(chord_freqs):
            start = i * int(r * 0.08)
            end = len(jt)
            seg_t = np.linspace(0, (end - start) / r, end - start, endpoint=False)
            env = np.exp(-seg_t * 3.5)
            jfanfare[start:end] += (
                np.sin(2 * np.pi * freq * seg_t) * 0.4
                + np.sin(2 * np.pi * freq * 2 * seg_t) * 0.15
                + np.sin(2 * np.pi * freq * 3 * seg_t) * 0.07
            ) * env
        # Add a quick drum-hit transient at the start
        drum_len = int(r * 0.05)
        rng = np.random.default_rng(42)
        drum = rng.uniform(-1, 1, drum_len) * np.exp(-np.linspace(0, 1, drum_len) * 30)
        jfanfare[:drum_len] += drum * 0.35
        jfanfare /= np.max(np.abs(jfanfare) + 1e-9)
        jfanfare *= 0.85
        self._save("jackpot_fanfare", jfanfare)

        # ── broke: descending "wah-wah" — plays when credits are reset silently
        # Three falling notes with a muted trombone-like tone
        broke = np.zeros(int(r * 0.9))
        wah_notes = [(466, 0.0), (370, 0.22), (277, 0.48)]   # Bb4 → F#4 → Db4
        for freq, t_start in wah_notes:
            start = int(r * t_start)
            dur   = int(r * 0.28)
            end   = min(start + dur, len(broke))
            seg_t = np.linspace(0, (end - start) / r, end - start, endpoint=False)
            env   = np.exp(-seg_t * 8) * (1 - np.exp(-seg_t * 60))  # soft attack
            # slightly detuned harmonics for a "wah" timbre
            sig = (
                np.sin(2 * np.pi * freq * seg_t) * 0.55
                + np.sin(2 * np.pi * freq * 1.5 * seg_t) * 0.20
                + np.sin(2 * np.pi * freq * 2.0 * seg_t) * 0.10
            ) * env
            broke[start:end] += sig
        broke /= np.max(np.abs(broke) + 1e-9)
        broke *= 0.65
        self._save("broke", broke)

    # ── Public play API ────────────────────────────────────────────────────

    def play(self, name: str):
        if not _NUMPY:
            return
        fx = self._effects.get(name)
        if fx:
            fx.play()

    def cleanup(self):
        import shutil
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass


# ─── Payout multiplier table ──────────────────────────────────────────────────
# Values are bet multipliers (win = catch_mult × bet).
# Full-catch rows on jackpot tiers (7–10) use multiplier 0 — the progressive pot pays instead.
# Calibrated for video-keno style fixed-pay RTP ≈ 72–92% (house edge ≈ 8–28%).
# Progressive 2% contribution + double-up 3× risk keep long-run entertainment edge healthy.
# 0-catch consolation on picks 7–10 improves hit frequency on higher tickets.
PAYOUTS = {
    1:  {1: 3},
    2:  {2: 12},
    3:  {3: 42,     2: 2},
    4:  {4: 130,    3: 5,    2: 1},
    5:  {5: 800,    4: 18,   3: 2},
    6:  {6: 1800,   5: 70,   4: 6,   3: 1},
    7:  {7: 0,      6: 280,  5: 22,  4: 3,  3: 1,  0: 1},
    8:  {8: 0,      7: 700,  6: 55,  5: 12, 4: 2,  0: 2},
    9:  {9: 0,      8: 2500, 7: 200, 6: 30, 5: 5,  4: 1, 0: 2},
    10: {10: 0,     9: 5000, 8: 600, 7: 100, 6: 18, 5: 3, 4: 1, 0: 3},
}

# ─── Progressive jackpot seeds & contribution rate ────────────────────────────
# Every pick count 1-10 has its own jackpot tier (full catch wins the pot).
# Seeds scale with pick difficulty — harder to hit = larger seed.
# Pots are stored and paid at face value regardless of current bet size.
# Contribution: 2% of each bet goes primarily to the *current* spots tier
# (70%) with the remainder (30%) shared across all other tiers.
JACKPOT_SEEDS = {
    # Seeds are set so the progressive is never below the best fixed prize
    # at the maximum bet (100). Settlement also enforces a hard floor.
    1:  500,
    2:  1_500,
    3:  5_000,
    4:  15_000,
    5:  100_000,
    6:  200_000,
    7:  40_000,
    8:  100_000,
    9:  300_000,
    10: 600_000,
}
JACKPOT_CONTRIB = 0.02          # fraction of each bet fed into the pots
JACKPOT_PRIMARY_SHARE = 0.70    # portion of contribution that goes to the active tier
# Time-based progressive drip (visual odometer + slow real growth between bets)
JACKPOT_TICK_MS = 3000          # interval between +1 ticks (~every 3 seconds)
JACKPOT_TICK_AMOUNT = 1         # credits added to each tier per tick
JACKPOT_TICK_SAVE_EVERY = 20    # persist to disk every N ticks (avoid I/O spam)


def jackpot_value_for_bet(pots: dict[int, int], tier: int, bet: int = 1) -> int:
    """Return the current jackpot value for a tier (face value, not bet-scaled)."""
    if tier not in JACKPOT_SEEDS:
        return 0
    return max(int(pots.get(tier, 0)), JACKPOT_SEEDS[tier])


_SAVE_PATH = Path.home() / ".keno_save.json"

DEFAULT_BALANCE = 1_000


def _load_save() -> dict:
    try:
        return json.loads(_SAVE_PATH.read_text())
    except Exception:
        return {}


def _load_jackpots() -> dict[int, int]:
    data = _load_save()
    pots = data.get("jackpots", {})
    try:
        return {int(k): max(int(v), JACKPOT_SEEDS[int(k)])
                for k, v in pots.items() if int(k) in JACKPOT_SEEDS}
    except Exception:
        return {}

def _load_balance() -> int:
    try:
        v = int(_load_save().get("balance", DEFAULT_BALANCE))
        return max(0, v)   # allow zero after a broke session; rebuy restores credits
    except Exception:
        return DEFAULT_BALANCE


def _save_state(pots: dict[int, int], balance: int) -> None:
    try:
        existing = _load_save()
        existing["jackpots"] = {str(k): v for k, v in pots.items()}
        # Persist any non-default balance (including 0 after broke)
        if balance != DEFAULT_BALANCE:
            existing["balance"] = balance
        else:
            existing.pop("balance", None)
        _SAVE_PATH.write_text(json.dumps(existing))
    except Exception:
        pass


# Keep old name as shim so any lingering internal calls still work
def _save_jackpots(pots: dict[int, int]) -> None:
    # balance unknown here; reload from disk to avoid clobbering it
    try:
        existing = _load_save()
        existing["jackpots"] = {str(k): v for k, v in pots.items()}
        _SAVE_PATH.write_text(json.dumps(existing))
    except Exception:
        pass



def format_credits(amount: int) -> str:
    """Format an integer amount without currency or credit suffixes."""
    return f"{amount:,}"


def format_winnings(amount: int) -> str:
    """Format winnings with a plus sign for the animated win indicator, without suffixes."""
    return f"+{amount:,}"


DRAW_INTERVAL_MS = 150
DRAWN_CHIP_SLIDE_MS = 220
DRAWN_CHIP_SIZE = 32
DRAWN_CHIP_SPACING = 4
DRAWN_CHIPS_COUNT = 20

# State → (fill_top, fill_bot, border, text, glow)
BALL_THEME = {
    "default": ("#0a1628", "#071020", "#1a3a6a", "#2a5a9a", None),
    "selected": ("#ffcc00", "#e08800", "#ffee44", "#000000", "#ffdd00"),
    "hit": ("#00dd55", "#009933", "#44ff88", "#ffffff", "#00ff66"),
    "drawn_miss": ("#3a0a0a", "#200505", "#cc2222", "#ff4444", "#dd0000"),
}


class BallWidget(QWidget):
    """Custom-painted circular keno ball."""

    def __init__(self, number: int, parent=None):
        super().__init__(parent)
        self.number = number
        self._state = "default"
        self.setFixedSize(66, 66)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cb = None

    def set_click_callback(self, cb):
        self._cb = cb

    def set_state(self, state: str):
        if self._state != state:
            self._state = state
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._cb:
            self._cb(self.number)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        fill_top, fill_bot, border_col, text_col, glow_col = BALL_THEME[self._state]
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        # Keep every glow ring fully inside the fixed 66×66 widget.
        # The previous math let the outer ring exceed the paint bounds, which clipped it.
        edge_pad = 2
        max_outer_radius = min(w, h) / 2 - edge_pad
        if glow_col:
            r = int(max_outer_radius - 10)
            # Draw from outermost to innermost. Radius + half pen width is capped
            # below max_outer_radius so antialiasing never gets cut off at the edge.
            for i in range(4, 0, -1):
                pen_w = i + 2
                rr = min(r + (i * 2), max_outer_radius - (pen_w / 2))
                c = QColor(glow_col)
                c.setAlpha(42 + (4 - i) * 26)
                p.setPen(QPen(c, pen_w))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(
                    int(cx - rr),
                    int(cy - rr),
                    int(rr * 2),
                    int(rr * 2),
                )
        else:
            r = min(w, h) // 2 - 4

        # Ball gradient fill
        grad = QRadialGradient(cx - r // 3, cy - r // 3, r * 1.4)
        grad.setColorAt(0.0, QColor(fill_top))
        grad.setColorAt(1.0, QColor(fill_bot))
        p.setBrush(QBrush(grad))
        border_c = QColor(border_col)
        border_c.setAlpha(220)
        p.setPen(QPen(border_c, 2))
        p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # Specular highlight
        if self._state in ("selected", "hit"):
            hi = QRadialGradient(cx - r // 3, cy - r // 2, r // 2)
            hi.setColorAt(0.0, QColor(255, 255, 255, 90))
            hi.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.setBrush(QBrush(hi))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        font_size = 13 if self.number < 10 else 12
        p.setFont(QFont("Arial Black", font_size, QFont.Weight.ExtraBold))
        p.setPen(QColor(text_col))
        p.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, str(self.number))
        p.end()


class GradientPanel(QWidget):
    def __init__(self, top="#0d1f3c", bottom="#060e1a", parent=None):
        super().__init__(parent)
        self._top = top
        self._bot = bottom

    def paintEvent(self, event):
        p = QPainter(self)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(self._top))
        grad.setColorAt(1, QColor(self._bot))
        p.fillRect(self.rect(), QBrush(grad))
        p.end()


class CabinetHeader(QWidget):
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, QColor("#0a1628"))
        grad.setColorAt(0.35, QColor("#1a3a7a"))
        grad.setColorAt(0.5, QColor("#2a5aaa"))
        grad.setColorAt(0.65, QColor("#1a3a7a"))
        grad.setColorAt(1.0, QColor("#0a1628"))
        p.fillRect(self.rect(), QBrush(grad))
        for y_off in (0, self.height() - 3):
            lg = QLinearGradient(0, 0, self.width(), 0)
            lg.setColorAt(0.0, QColor("#0a1628"))
            lg.setColorAt(0.2, QColor("#ccaa00"))
            lg.setColorAt(0.5, QColor("#ffdd44"))
            lg.setColorAt(0.8, QColor("#ccaa00"))
            lg.setColorAt(1.0, QColor("#0a1628"))
            p.fillRect(0, y_off, self.width(), 3, QBrush(lg))
        p.setFont(QFont("Arial Black", 26, QFont.Weight.ExtraBold))
        p.setPen(QColor("#000000"))
        p.drawText(
            QRect(2, 12, self.width(), 50),
            Qt.AlignmentFlag.AlignHCenter,
            "SUPER DOUBLE UP",
        )
        p.setPen(QColor("#ffdd33"))
        p.drawText(
            QRect(0, 10, self.width(), 50),
            Qt.AlignmentFlag.AlignHCenter,
            "SUPER DOUBLE UP",
        )
        p.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        p.setPen(QColor("#88bbff"))
        p.drawText(
            QRect(0, 56, self.width(), 22),
            Qt.AlignmentFlag.AlignHCenter,
            "✦  V I D E O   K E N O  ✦",
        )
        p.end()


class BottomPanel(QWidget):
    def paintEvent(self, event):
        p = QPainter(self)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor("#0d1f3c"))
        grad.setColorAt(1, QColor("#060e1a"))
        p.fillRect(self.rect(), QBrush(grad))
        lg = QLinearGradient(0, 0, self.width(), 0)
        lg.setColorAt(0.0, QColor("#0a1628"))
        lg.setColorAt(0.15, QColor("#ccaa00"))
        lg.setColorAt(0.5, QColor("#ffdd44"))
        lg.setColorAt(0.85, QColor("#ccaa00"))
        lg.setColorAt(1.0, QColor("#0a1628"))
        p.fillRect(0, 0, self.width(), 3, QBrush(lg))
        p.end()


class DigitalLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Courier New", 15, QFont.Weight.Bold))
        self.setStyleSheet("""
            color: #ffdd33; background: #000a14;
            border: 2px solid #1a4a7a; border-radius: 4px; padding: 4px 12px;
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)




class RollingDrawnBall(QWidget):
    """Painted drawn-number ball that can rotate while it slides into the row."""

    def __init__(self, number: int, is_hit: bool, parent=None):
        super().__init__(parent)
        self.number = number
        self.is_hit = is_hit
        self._angle = 0.0
        self.setFixedSize(DRAWN_CHIP_SIZE, DRAWN_CHIP_SIZE)

    def get_angle(self):
        return self._angle

    def set_angle(self, angle):
        self._angle = float(angle)
        self.update()

    angle = pyqtProperty(float, fget=get_angle, fset=set_angle)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = min(self.width(), self.height())
        radius = size // 2 - 2
        cx = self.width() // 2
        cy = self.height() // 2

        p.translate(cx, cy)
        p.rotate(self._angle)
        p.translate(-cx, -cy)

        if self.is_hit:
            fill_top, fill_bot = "#00ee66", "#008833"
            border = QColor("#44ff88")
            glow_col = QColor("#00ff66")
            text = QColor("#ffffff")
            shine_alpha = 95
        else:
            fill_top, fill_bot = "#3a0a0a", "#200505"
            border = QColor("#cc2222")
            glow_col = QColor("#dd0000")
            text = QColor("#ff4444")
            shine_alpha = 70

        # Glow rings (mirrors BallWidget logic)
        edge_pad = 2
        max_outer_radius = min(self.width(), self.height()) / 2 - edge_pad
        r = int(max_outer_radius - 4)
        for i in range(4, 0, -1):
            pen_w = i + 2
            rr = min(r + (i * 2), max_outer_radius - (pen_w / 2))
            c = QColor(glow_col)
            c.setAlpha(42 + (4 - i) * 26)
            p.setPen(QPen(c, pen_w))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(int(cx - rr), int(cy - rr), int(rr * 2), int(rr * 2))

        grad = QRadialGradient(cx - radius // 3, cy - radius // 3, radius * 1.4)
        grad.setColorAt(0.0, QColor(fill_top))
        grad.setColorAt(1.0, QColor(fill_bot))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(border, 2 if self.is_hit else 1))
        p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        highlight = QRadialGradient(cx - radius // 3, cy - radius // 2, radius // 2)
        highlight.setColorAt(0.0, QColor(255, 255, 255, shine_alpha))
        highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(highlight))
        p.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        p.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        p.setPen(text)
        p.drawText(QRect(0, 0, self.width(), self.height()), Qt.AlignmentFlag.AlignCenter, str(self.number))
        p.end()



class DmdPanel(QWidget):
    """Shared RGB dot-matrix display renderer.  Subclasses use the protected helpers.

    Palette is multi-colour (true RGB LEDs): cyan labels, green values,
    amber highlights, gold jackpots, red alerts — not monochrome.
    """

    # Semantic RGB LED colours used across all cabinet matrices
    RGB = {
        "red":     "#ff3355",
        "green":   "#00ff66",
        "blue":    "#3a8cff",
        "cyan":    "#00e8ff",
        "amber":   "#ffaa00",
        "gold":    "#ffcc33",
        "white":   "#eef5ff",
        "magenta": "#ff44cc",
        "dim":     "#1a2838",
    }

    @classmethod
    def _rgb(cls, name: str) -> "QColor":
        return QColor(cls.RGB.get(name, "#ffffff"))

    _GLYPHS = {
        "0": ("111","101","101","101","101","101","111"),
        "1": ("010","110","010","010","010","010","111"),
        "2": ("111","001","001","111","100","100","111"),
        "3": ("111","001","001","111","001","001","111"),
        "4": ("101","101","101","111","001","001","001"),
        "5": ("111","100","100","111","001","001","111"),
        "6": ("111","100","100","111","101","101","111"),
        "7": ("111","001","001","010","010","010","010"),
        "8": ("111","101","101","111","101","101","111"),
        "9": ("111","101","101","111","001","001","111"),
        ",": ("0","0","0","0","0","1","1"),
        "+": ("000","010","010","111","010","010","000"),
        "-": ("000","000","000","111","000","000","000"),
        "=": ("000","000","111","000","111","000","000"),
        " ": ("0","0","0","0","0","0","0"),
        "A": ("111","101","101","111","101","101","101"),
        "B": ("110","101","101","110","101","101","110"),
        "C": ("111","100","100","100","100","100","111"),
        "D": ("110","101","101","101","101","101","110"),
        "E": ("111","100","100","111","100","100","111"),
        "F": ("111","100","100","111","100","100","100"),
        "G": ("111","100","100","101","101","101","111"),
        "H": ("101","101","101","111","101","101","101"),
        "I": ("111","010","010","010","010","010","111"),
        "J": ("001","001","001","001","101","101","111"),
        "K": ("101","101","110","100","110","101","101"),
        "L": ("100","100","100","100","100","100","111"),
        "M": ("101","111","111","101","101","101","101"),
        "N": ("101","111","111","101","101","101","101"),
        "O": ("111","101","101","101","101","101","111"),
        "P": ("111","101","101","111","100","100","100"),
        "R": ("111","101","101","111","110","101","101"),
        "S": ("111","100","100","111","001","001","111"),
        "T": ("111","010","010","010","010","010","010"),
        "U": ("101","101","101","101","101","101","111"),
        "V": ("101","101","101","101","101","010","010"),
        "W": ("101","101","101","101","101","111","101"),
        "X": ("101","101","010","010","010","101","101"),
        "Y": ("101","101","101","111","010","010","010"),
        "Z": ("111","001","001","010","100","100","111"),
        "/": ("001","001","010","010","010","100","100"),
        ".": ("0","0","0","0","0","0","1"),
        ":": ("0","1","0","0","0","1","0"),
    }

    _IDLE_MS = 50;  _ROLL_MS = 30;  _FLICKER_MS = 40
    _CHASE_MS = 90; _AFTERGLOW_MS = 50; _ROLL_STEPS = 12

    def __init__(self, dmd_cols=104, dmd_rows=32,
                 border_col="#442400", bg_col="#050302", recess_col="#100500",
                 parent=None):
        super().__init__(parent)
        self._dmd_cols   = dmd_cols
        self._dmd_rows   = dmd_rows
        self._border_col = border_col
        self._bg_col     = bg_col
        self._recess_col = recess_col
        self._breathe_phase: float = 0.0
        self._scanline_y:    float = 0.0
        self._idle_timer = QTimer(self)
        self._idle_timer.timeout.connect(self._tick_idle_base)
        self._idle_timer.start(self._IDLE_MS)

    def _tick_idle_base(self):
        self._breathe_phase = (self._breathe_phase + 0.04) % (2 * math.pi)
        self._scanline_y    = (self._scanline_y    + 0.004) % 1.0
        self.update()

    def _breathe(self) -> float:
        return 0.75 + 0.25 * (0.5 + 0.5 * math.sin(self._breathe_phase))

    def _grid_params(self, W, H):
        rx, ry, rw, rh = 7, 7, W - 14, H - 14
        cell_w = (rw - 10) / self._dmd_cols
        cell_h = (rh - 10) / self._dmd_rows
        pitch  = min(cell_w, cell_h)
        dot    = max(1.0, pitch * 0.70)
        gx0    = rx + (rw - self._dmd_cols * pitch) / 2
        gy0    = ry + (rh - self._dmd_rows * pitch) / 2
        return rx, ry, rw, rh, pitch, dot, gx0, gy0

    def _paint_base(self, p, W, H, off_col, on_col=None,
                    border_override=None, scanline=True):
        """Draw housing + recess + all dim dots + optional scanline.
        Returns (pitch, dot, gx0, gy0)."""
        bc = border_override or QColor(self._border_col)
        p.setBrush(QBrush(QColor(self._bg_col)))
        p.setPen(QPen(bc, 2))
        p.drawRoundedRect(1, 1, W-2, H-2, 5, 5)

        rx, ry, rw, rh, pitch, dot, gx0, gy0 = self._grid_params(W, H)
        p.setBrush(QBrush(QColor(self._recess_col)))
        p.setPen(QPen(QColor("#2a1200"), 1))
        p.drawRoundedRect(rx, ry, rw, rh, 3, 3)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(off_col))
        for row in range(self._dmd_rows):
            for col in range(self._dmd_cols):
                x = gx0 + col * pitch
                y = gy0 + row * pitch
                p.drawEllipse(int(x), int(y), max(1,int(dot)), max(1,int(dot)))

        if scanline and on_col:
            sy_abs   = gy0 + self._scanline_y * (self._dmd_rows * pitch)
            sh_px    = pitch * 2.5
            for row in range(self._dmd_rows):
                y    = gy0 + row * pitch
                dist = abs(y - sy_abs)
                if dist < sh_px:
                    alpha = int(28 * (1.0 - dist / sh_px))
                    for col in range(self._dmd_cols):
                        x  = gx0 + col * pitch
                        sc = QColor(on_col); sc.setAlpha(alpha)
                        p.setBrush(QBrush(sc))
                        p.drawEllipse(int(x), int(y), max(1,int(dot)), max(1,int(dot)))
        return pitch, dot, gx0, gy0

    @classmethod
    def _scale_bri(cls, base: QColor, bri: float) -> QColor:
        c = QColor(base); h,s,v,a = c.getHsvF()
        c.setHsvF(h, s, min(1.0, v*bri), a); return c

    @classmethod
    def _text_width(cls, text: str, scale: int, gap: int) -> int:
        w = 0
        for ch in text.upper():
            g = cls._GLYPHS.get(ch, cls._GLYPHS[" "])
            w += len(g[0]) * scale + gap
        return max(0, w - gap)

    @classmethod
    def _draw_dmd_text(cls, painter, text, gx0, gy0, pitch,
                       start_col, start_row, scale, gap, dot,
                       on_col, glow_col, max_col=99999, max_row=99999):
        col_cursor = start_col
        for ch in text.upper():
            glyph = cls._GLYPHS.get(ch, cls._GLYPHS[" "])
            gw    = len(glyph[0])
            for gy, row_bits in enumerate(glyph):
                for gx, bit in enumerate(row_bits):
                    if bit != "1": continue
                    for sy in range(scale):
                        for sx in range(scale):
                            col = col_cursor + gx*scale + sx
                            row = start_row  + gy*scale + sy
                            if col > max_col or row > max_row: continue
                            x = gx0 + col*pitch
                            y = gy0 + row*pitch
                            gr = QRadialGradient(x+dot/2, y+dot/2, dot*2.4)
                            gr.setColorAt(0.0, glow_col)
                            gr.setColorAt(1.0, QColor(0,0,0,0))
                            painter.setBrush(QBrush(gr))
                            painter.setPen(Qt.PenStyle.NoPen)
                            painter.drawEllipse(int(x-dot*0.5), int(y-dot*0.5),
                                                max(1,int(dot*2.0)), max(1,int(dot*2.0)))
                            painter.setBrush(QBrush(on_col))
                            painter.drawEllipse(int(x), int(y),
                                                max(1,int(dot)), max(1,int(dot)))
            col_cursor += gw*scale + gap

    def _draw_divider(self, p, gx0, gy0, pitch, dot, row, on_col):
        """Draw a full-width row of dim dots as a visual divider."""
        dim = QColor(on_col); dim.setAlpha(60)
        for col in range(self._dmd_cols):
            x = gx0 + col * pitch
            y = gy0 + row * pitch
            p.setBrush(QBrush(dim)); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(int(x), int(y), max(1,int(dot)), max(1,int(dot)))



class JackpotStrip(DmdPanel):
    """RGB progressive jackpot matrix — gold value, cyan label, pulse on drip.

    Fixed 56×24 grid in a 185×78 panel. Two bands only so glyphs never clip:
      rows 2–8   : label  (e.g. "10-SPOT JP")  cyan  scale-1  (7 rows)
      rows 12–23 : value  (e.g. "600,000")     gold  scale-2  (14 rows)
    Scale auto-drops to 1 if a string is too wide.
    """

    def __init__(self, parent=None):
        super().__init__(
            dmd_cols=56, dmd_rows=26,
            border_col="#4a3800", bg_col="#080600", recess_col="#100a00",
            parent=parent,
        )
        self.setFixedWidth(185)
        self.setFixedHeight(84)
        self._spots = 0
        self._value = 0
        self._display_value = 0
        self._flash = False
        self._tick_pulse = False
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)
        self._tick_timer = QTimer(self)
        self._tick_timer.setSingleShot(True)
        self._tick_timer.timeout.connect(self._end_tick)
        self._roll_timer = QTimer(self)
        self._roll_timer.timeout.connect(self._tick_roll)

    def set_jackpot(self, spots: int, value: int, flash: bool = False, tick: bool = False):
        self._spots = spots
        changed = value != self._value
        self._value = value
        if flash:
            self._flash = True
            self._tick_pulse = False
            self._flash_timer.start(900)
            self._display_value = value
        elif tick:
            self._tick_pulse = True
            self._tick_timer.start(320)
            if changed and not self._roll_timer.isActive():
                self._roll_timer.start(30)
        else:
            if abs(value - self._display_value) > 50:
                if not self._roll_timer.isActive():
                    self._roll_timer.start(30)
            else:
                self._display_value = value
        self.update()

    def _end_flash(self):
        self._flash = False
        self.update()

    def _end_tick(self):
        self._tick_pulse = False
        self.update()

    def _tick_roll(self):
        diff = self._value - self._display_value
        if diff == 0:
            self._roll_timer.stop()
            return
        step = max(1, abs(diff) // 8)
        if diff > 0:
            self._display_value = min(self._value, self._display_value + step)
        else:
            self._display_value = max(self._value, self._display_value - step)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        bri = self._breathe()

        gold  = self._rgb("gold")
        cyan  = self._rgb("cyan")
        white = self._rgb("white")
        amber = self._rgb("amber")

        if self._flash:
            on_val, glow_val = white, QColor(255, 240, 180, 200)
            on_lbl, glow_lbl = white, QColor(180, 220, 255, 160)
            off, border = QColor("#2a2000"), QColor("#ffdd55")
        elif self._tick_pulse:
            on_val = self._scale_bri(gold, 1.15)
            glow_val = QColor(255, 200, 60, 150)
            on_lbl = self._scale_bri(cyan, 1.05)
            glow_lbl = QColor(0, 200, 255, 100)
            off, border = QColor("#1a1400"), QColor("#aa8800")
        else:
            on_val = self._scale_bri(gold, bri)
            glow_val = QColor(255, 180, 40, int(60 + 50 * bri))
            on_lbl = self._scale_bri(cyan, bri * 0.85)
            glow_lbl = QColor(0, 180, 220, int(40 + 30 * bri))
            off, border = QColor("#120e00"), None

        pitch, dot, gx0, gy0 = self._paint_base(
            p, W, H, off, on_col=on_val, border_override=border, scanline=True
        )
        cols, rows = self._dmd_cols, self._dmd_rows
        max_c, max_r = cols - 1, rows - 1

        def blit(text, row, sc, oc, gc, gap=1):
            # Shrink until horizontal + vertical fit
            while sc >= 1:
                h = 7 * sc
                w = self._text_width(text, sc, gap)
                if w <= cols - 2 and row + h - 1 <= max_r:
                    break
                sc -= 1
            if sc < 1:
                sc = 1
            w = self._text_width(text, sc, gap)
            col = max(0, (cols - min(w, cols - 2)) // 2)
            self._draw_dmd_text(
                p, text, gx0, gy0, pitch, col, row, sc, gap, dot,
                oc, gc, max_col=max_c, max_row=max_r,
            )

        if self._spots > 0:
            # Compact label — always fits at scale 1 in 56 cols
            blit(f"{self._spots}-SPOT JP", 2, 1, on_lbl, glow_lbl)
            blit(format_credits(self._display_value), 11, 2, on_val, glow_val)
        else:
            blit("PROGRESSIVE", 2, 1, on_lbl, glow_lbl)
            blit("PICK SPOTS", 12, 1, self._scale_bri(amber, bri * 0.75), glow_lbl)
        p.end()



class GameStatePanel(DmdPanel):
    """Amber DMD panel — always visible, always shows game state.

    IDLE   — large balance display + game title
    DRAW   — hit-dot tracker (one dot per player pick, lights up on each match)
    RESULT — WIN + amount (green) or NO WIN (dim) for 2.5 s, then reverts to idle
    """

    def __init__(self, parent=None):
        super().__init__(
            dmd_cols=104, dmd_rows=32,
            border_col="#1a2a3a", bg_col="#040a12", recess_col="#060e18",
            parent=parent,
        )
        self.setFixedWidth(185)
        self.setFixedHeight(104)

        # ── Shared state ─────────────────────────────────────────────────
        self._balance:   int = 0
        self._spots:     int = 0    # number of picks this round
        self._hits:      int = 0    # hits so far (draw mode)
        self._hit_list:  list[bool] = []   # ordered: True = hit, False = miss

        # ── Mode ─────────────────────────────────────────────────────────
        self._mode: str = "idle"   # "idle" | "draw" | "result"

        # ── Result mode ──────────────────────────────────────────────────
        self._result_win:    int  = 0
        self._result_timer = QTimer(self)
        self._result_timer.setSingleShot(True)
        self._result_timer.timeout.connect(self._end_result)

        # ── Flicker (used in result-win mode) ────────────────────────────
        self._flicker_phase: float = 0.0
        self._flicker_timer = QTimer(self)
        self._flicker_timer.timeout.connect(self._tick_flicker)

        # ── Balance odometer ─────────────────────────────────────────────
        self._bal_displayed: int = 0
        self._digit_states:  list[list] = []
        self._roll_timer = QTimer(self)
        self._roll_timer.timeout.connect(self._tick_roll)

    # ── Public API ─────────────────────────────────────────────────────────

    def set_balance(self, balance: int):
        if balance != self._balance:
            self._balance = balance
            if self._mode == "idle":
                self._start_roll(balance)
            else:
                self._set_instant(balance)

    def start_draw(self, spots: int):
        self._mode    = "draw"
        self._spots   = spots
        self._hits    = 0
        self._hit_list = []
        self._result_timer.stop()
        self._flicker_timer.stop()
        self.update()

    def register_hit(self, is_hit: bool):
        """Call for every ball drawn (hit or miss) to update the dot tracker."""
        self._hit_list.append(is_hit)
        if is_hit:
            self._hits += 1
        self.update()

    def show_result(self, winnings: int):
        self._mode       = "result"
        self._result_win = winnings
        self._result_timer.start(2500)
        if winnings > 0:
            self._flicker_phase = 0.0
            self._flicker_timer.start(self._FLICKER_MS)
        self.update()

    def stop_draw(self):
        if self._mode == "draw":
            self._mode = "idle"
            self._set_instant(self._balance)
            self.update()

    def _end_result(self):
        self._mode = "idle"
        self._flicker_timer.stop()
        self._set_instant(self._balance)
        self.update()

    # ── Tick handlers ──────────────────────────────────────────────────────

    def _tick_idle_base(self):
        self._breathe_phase = (self._breathe_phase + 0.04) % (2 * math.pi)
        self._scanline_y    = (self._scanline_y    + 0.004) % 1.0
        if self._mode == "idle":
            self.update()

    def _tick_flicker(self):
        self._flicker_phase = (self._flicker_phase + 0.18) % (2 * math.pi)
        if self._mode == "result":
            self.update()

    def _tick_roll(self):
        DIGITS = "0123456789"
        still  = False
        for state in self._digit_states:
            disp, target, step = state
            if disp == target:
                continue
            still    = True
            state[2] += 1
            if state[2] >= self._ROLL_STEPS or disp not in DIGITS:
                state[0] = target; state[2] = 0
            else:
                state[0] = DIGITS[(DIGITS.index(disp) + 1) % len(DIGITS)]
        self.update()
        if not still:
            self._roll_timer.stop()
            self._bal_displayed = self._balance

    # ── Internal helpers ───────────────────────────────────────────────────

    def _set_instant(self, value: int):
        self._bal_displayed = value
        text = format_credits(value)
        self._digit_states = [[c, c, 0] for c in text]
        self._roll_timer.stop()

    def _start_roll(self, new_value: int):
        old = format_credits(self._bal_displayed)
        new = format_credits(new_value)
        diff = len(new) - len(old)
        if diff > 0:   old = " " * diff + old
        elif diff < 0: new = " " * (-diff) + new
        self._digit_states = [[o, n, 0] for o, n in zip(old, new)]
        if not self._roll_timer.isActive():
            self._roll_timer.start(self._ROLL_MS)

    def _disp_text(self) -> str:
        return "".join(s[0] for s in self._digit_states) if self._digit_states \
               else format_credits(self._balance)

    def _bri(self) -> float:
        if self._mode == "result" and self._result_win > 0:
            return 0.70 + 0.30 * (0.5 + 0.5 * math.sin(self._flicker_phase))
        return self._breathe()

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        bri  = self._bri()

        # RGB palette
        cyan   = self._scale_bri(self._rgb("cyan"),   bri)
        green  = self._scale_bri(self._rgb("green"),  bri)
        amber  = self._scale_bri(self._rgb("amber"),  bri)
        red    = self._scale_bri(self._rgb("red"),    bri)
        blue   = self._scale_bri(self._rgb("blue"),   bri)
        white  = self._scale_bri(self._rgb("white"),  bri)
        off_col = QColor("#0a1218")
        # Default "on" is cyan for chrome, values use green
        on_col, glow_col = cyan, QColor(0, 180, 220, int(50 + 40 * bri))
        dim_on = self._scale_bri(self._rgb("cyan"), bri * 0.35)
        dim_glow = QColor(0, 100, 140, 40)
        val_on, val_glow = green, QColor(0, 220, 80, int(60 + 50 * bri))
        lbl_on, lbl_glow = blue, QColor(40, 100, 200, 50)

        pitch, dot, gx0, gy0 = self._paint_base(
            p, W, H, off_col, on_col,
            scanline=(self._mode == "idle"))
        cols = self._dmd_cols   # 104
        rows = self._dmd_rows   # 32

        def txt(text, col, row, sc=1, gap=1, oc=None, gc=None):
            self._draw_dmd_text(p, text, gx0, gy0, pitch, col, row,
                                sc, gap, dot, oc or on_col, gc or glow_col)

        def centre(text, row, sc=1, gap=1, oc=None, gc=None):
            w = self._text_width(text, sc, gap)
            while sc > 1 and w > cols - 2:
                sc -= 1
                w = self._text_width(text, sc, gap)
            txt(text, max(0, (cols - w) // 2), row, sc, gap, oc, gc)

        def divider(row):
            self._draw_divider(p, gx0, gy0, pitch, dot, row, dim_on)

        # ──────────────────────────────────────────────────────────────────
        if self._mode == "idle":
            centre("SUPER KENO", 2, oc=cyan, gc=glow_col)
            divider(10)
            bal_str = self._disp_text().lstrip()
            bw = self._text_width(bal_str, 2, 1)
            while 2 > 1 and bw > cols - 2:
                break  # scale-2 is intentional for balance
            txt(bal_str, max(0, (cols - bw) // 2), 13, 2, 1, val_on, val_glow)
            centre("CREDITS", 27, oc=lbl_on, gc=lbl_glow)

        # ──────────────────────────────────────────────────────────────────
        elif self._mode == "draw":
            centre("DRAWN BALLS", 2, oc=cyan, gc=glow_col)
            divider(10)

            # Two rows of 10 dots — all 20 drawn balls, in draw order.
            # Hits glow amber, misses are dim rings, pending are tiny dots.
            total_slots = 20
            per_row     = 10
            margin_frac = 0.05
            usable_w    = cols * pitch * (1 - 2 * margin_frac)
            r_dot       = max(2.5, min(5.0, usable_w / (per_row * 2.4)))
            spacing     = usable_w / per_row
            x_start     = gx0 + cols * pitch * margin_frac + spacing * 0.5
            row_y       = [gy0 + 14 * pitch, gy0 + 22 * pitch]

            for i in range(total_slots):
                row_idx = i // per_row
                col_idx = i  % per_row
                cx = x_start + col_idx * spacing
                cy = row_y[row_idx] + r_dot

                if i < len(self._hit_list):
                    lit = self._hit_list[i]
                else:
                    lit = None

                if lit is True:
                    # Hit = bright green
                    gcol = QColor(0, 255, 100, 180)
                    gr = QRadialGradient(cx, cy, r_dot * 2.5)
                    gr.setColorAt(0.0, gcol); gr.setColorAt(1.0, QColor(0,0,0,0))
                    p.setBrush(QBrush(gr)); p.setPen(Qt.PenStyle.NoPen)
                    p.drawEllipse(int(cx-r_dot*2), int(cy-r_dot*2),
                                  int(r_dot*4), int(r_dot*4))
                    p.setBrush(QBrush(green)); p.setPen(Qt.PenStyle.NoPen)
                    p.drawEllipse(int(cx-r_dot), int(cy-r_dot),
                                  int(r_dot*2), int(r_dot*2))
                elif lit is False:
                    # Miss = dim red ring
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(self._scale_bri(self._rgb("red"), 0.45), max(1, int(r_dot * 0.35))))
                    p.drawEllipse(int(cx-r_dot), int(cy-r_dot),
                                  int(r_dot*2), int(r_dot*2))
                else:
                    # Pending = dim cyan speck
                    p.setBrush(QBrush(dim_on)); p.setPen(Qt.PenStyle.NoPen)
                    p.drawEllipse(int(cx-r_dot*0.4), int(cy-r_dot*0.4),
                                  int(r_dot*0.8), int(r_dot*0.8))

            divider(27)
            hits_str = f"{self._hits} HITS"
            hw = self._text_width(hits_str, 1, 1)
            hit_col = green if self._hits > 0 else dim_on
            txt(hits_str, max(0, (cols - hw) // 2), 29, oc=hit_col)

        # ──────────────────────────────────────────────────────────────────
        elif self._mode == "result":
            if self._result_win > 0:
                # Flicker-boosted green win
                pulse = 0.85 + 0.15 * abs(__import__("math").sin(self._flicker_phase))
                win_on   = self._scale_bri(self._rgb("green"), pulse)
                win_glow = QColor(0, 255, 100, int(120 * pulse))
                centre("WIN", 2, oc=win_on, gc=win_glow)
                divider(10)
                win_str = format_credits(self._result_win)
                ww = self._text_width(win_str, 2, 1)
                txt(win_str, max(0, (cols - ww) // 2), 13, 2, 1, win_on, win_glow)
                centre("CREDITS", 27, oc=lbl_on, gc=lbl_glow)
            else:
                centre("NO WIN", rows // 2 - 3, oc=self._scale_bri(self._rgb("red"), 0.55),
                       gc=QColor(180, 40, 40, 60))

        p.end()

class PayoutDMD(DmdPanel):
    """Green dot-matrix display replacing the payout table.

    Three display modes:
      IDLE   — payout table (catch + scaled prize, right-justified) + picks/bet header
      DRAW   — scrolling prize list; currently-reachable row highlighted amber
      RESULT — session stats (rounds, total won, biggest win) for 3 s, then back to idle
    """

    # Scroll tick for DRAW mode prize list
    _SCROLL_MS = 600

    def __init__(self, parent=None):
        super().__init__(
            dmd_cols=104, dmd_rows=96,
            border_col="#1a3a2a", bg_col="#020a08", recess_col="#041208",
            parent=parent,
        )
        self.setFixedWidth(185)

        # ── Current game state ───────────────────────────────────────────
        self._spots:        int  = 0
        self._bet:          int  = 1
        self._prize_mult:   int  = 1
        self._jackpot_val:  int  = 0
        self._hits:         int  = -1
        self._winner_hits:  int  = -1
        self._balance:      int  = 0
        self._draws_done:   int  = 0
        self._draws_total:  int  = 20

        # ── DRAW mode ────────────────────────────────────────────────────
        self._in_draw: bool = False

        # ── RESULT mode (session stats) ──────────────────────────────────
        self._in_result:   bool = False
        self._sess_rounds: int  = 0
        self._sess_total:  int  = 0
        self._sess_best:   int  = 0
        self._last_hits:   int  = 0
        self._last_spots:  int  = 0
        self._last_win:    int  = 0
        self._result_timer = QTimer(self)
        self._result_timer.setSingleShot(True)
        self._result_timer.timeout.connect(self._end_result)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(200)

    # ── Public API ─────────────────────────────────────────────────────────

    def update_spots(self, spots, hits=-1, winner_hits=-1,
                     prize_multiplier=1, jackpot_value=0, bet=1):
        self._spots       = spots
        self._bet         = max(1, bet)
        self._prize_mult  = max(1, prize_multiplier)
        self._jackpot_val = jackpot_value
        self._hits        = hits
        self._winner_hits = winner_hits
        if not self._in_draw and not self._in_result:
            self.update()

    def set_balance(self, balance: int):
        self._balance = balance
        self.update()

    def start_draw(self, spots, bet, prize_mult, jackpot_val, balance=0):
        self._spots       = spots
        self._bet         = max(1, bet)
        self._prize_mult  = max(1, prize_mult)
        self._jackpot_val = jackpot_val
        self._hits        = 0
        self._winner_hits = -1
        self._balance     = balance
        self._draws_done  = 0
        self._in_draw     = True
        self._in_result   = False
        self.update()

    def update_draw(self, hits: int, draws_done: int = 0):
        self._hits       = hits
        self._draws_done = draws_done
        if self._in_draw:
            self.update()

    def stop_draw(self):
        self._in_draw = False
        self.update()

    def show_result(self, rounds: int, total_won: int, best_win: int,
                    last_hits: int = 0, last_spots: int = 0, last_win: int = 0):
        self._in_result   = True
        self._in_draw     = False
        self._sess_rounds = rounds
        self._sess_total  = total_won
        self._sess_best   = best_win
        self._last_hits   = last_hits
        self._last_spots  = last_spots
        self._last_win    = last_win
        self._result_timer.start(3500)
        self.update()

    def _end_result(self):
        self._in_result = False
        self.update()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _prize_rows(self) -> list[tuple[int, int]]:
        table = PAYOUTS.get(self._spots, {})
        rows  = []
        for catch in sorted(table.keys(), reverse=True):
            if catch == self._spots and self._spots in JACKPOT_SEEDS:
                prize = self._jackpot_val or JACKPOT_SEEDS[self._spots]
            else:
                prize = table[catch] * self._bet * self._prize_mult
            rows.append((catch, prize))
        return rows

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        bri  = self._breathe()

        # RGB semantic colours
        green  = self._scale_bri(self._rgb("green"),  bri)
        cyan   = self._scale_bri(self._rgb("cyan"),   bri)
        amber  = self._scale_bri(self._rgb("amber"),  bri)
        gold   = self._scale_bri(self._rgb("gold"),   bri)
        red    = self._scale_bri(self._rgb("red"),    bri)
        blue   = self._scale_bri(self._rgb("blue"),   bri)
        white  = self._scale_bri(self._rgb("white"),  bri)
        off_col = QColor("#06100a")
        on_col, glow_col = green, QColor(0, 200, 80, int(50 + 40 * bri))
        hi_on, hi_glow = amber, QColor(255, 160, 0, int(70 + 40 * bri))
        win_on, win_glow = white, QColor(100, 255, 160, 180)
        dim_on = self._scale_bri(self._rgb("green"), bri * 0.38)
        dim_glow = QColor(0, 100, 50, 40)
        lbl_on = self._scale_bri(self._rgb("cyan"), bri * 0.75)
        lbl_glow = QColor(0, 160, 200, 50)

        pitch, dot, gx0, gy0 = self._paint_base(
            p, W, H, off_col, on_col, scanline=not self._in_result)
        cols = self._dmd_cols
        rows = self._dmd_rows   # 96

        def txt(text, col, row, sc=1, gap=1, oc=None, gc=None):
            self._draw_dmd_text(p, text, gx0, gy0, pitch, col, row,
                                sc, gap, dot, oc or on_col, gc or glow_col)

        def centre(text, row, sc=1, gap=1, oc=None, gc=None):
            w = self._text_width(text, sc, gap)
            while sc > 1 and w > cols - 2:
                sc -= 1
                w = self._text_width(text, sc, gap)
            txt(text, max(0, (cols - w) // 2), row, sc, gap, oc, gc)

        def right(text, row, sc=1, gap=1, oc=None, gc=None):
            w = self._text_width(text, sc, gap)
            while sc > 1 and w > cols - 6:
                sc -= 1
                w = self._text_width(text, sc, gap)
            txt(text, max(0, cols - w - 4), row, sc, gap, oc, gc)

        def divider(row):
            self._draw_divider(p, gx0, gy0, pitch, dot, row, dim_on)

        def draw_prize_table(prize_rows, start_row, end_row):
            n = len(prize_rows)
            if n == 0:
                return
            available = end_row - start_row
            step = max(8, available // n)
            for slot, (catch, prize) in enumerate(prize_rows):
                dr = start_row + slot * step
                if dr + 7 > end_row:
                    break
                if catch == self._winner_hits:
                    co, cg = win_on, win_glow          # won row — white
                elif catch == self._hits and self._hits >= 0:
                    co, cg = hi_on, hi_glow            # current reach — amber
                else:
                    co, cg = amber, hi_glow            # catch # — amber
                txt(str(catch), 4, dr, oc=co, gc=cg)
                p_str = format_credits(prize)
                pw = self._text_width(p_str, 1, 1)
                # Prize amount always green (unless highlighted)
                po = win_on if catch == self._winner_hits else (
                    hi_on if catch == self._hits and self._hits >= 0 else green)
                pg = win_glow if catch == self._winner_hits else glow_col
                txt(p_str, max(0, cols - pw - 4), dr, oc=po, gc=pg)

        # ──────────────────────────────────────────────────────────────────
        if self._in_result:
            centre("SESSION STATS", 1, oc=cyan, gc=lbl_glow)
            divider(9)

            stat_area_start = 11
            stat_area_end   = rows - 14
            stat_h          = (stat_area_end - stat_area_start) // 3
            stats = [
                ("ROUNDS", str(self._sess_rounds)),
                ("TOTAL",  format_credits(self._sess_total)),
                ("BEST",   format_credits(self._sess_best)),
            ]
            for i, (lbl, val) in enumerate(stats):
                dr = stat_area_start + i * stat_h
                txt(lbl, 2, dr, 1, 1, lbl_on, lbl_glow)
                right(val, dr + 2, 2, 1, green if i == 0 else (gold if i == 2 else amber),
                      glow_col if i == 0 else hi_glow)
                if i < 2:
                    divider(dr + stat_h - 1)

            divider(rows - 12)
            if self._last_spots > 0:
                result_str = f"CATCH {self._last_hits}/{self._last_spots}"
                if self._last_win > 0:
                    centre(result_str, rows - 10, oc=win_on, gc=win_glow)
                    centre(f"+{format_credits(self._last_win)}", rows - 3, oc=hi_on, gc=hi_glow)
                else:
                    centre(result_str, rows - 8, oc=dim_on, gc=dim_glow)

        # ──────────────────────────────────────────────────────────────────
        elif self._in_draw:
            hdr = f"PICK {self._spots}  BET {self._bet}"
            if self._prize_mult > 1:
                hdr += f"  X{self._prize_mult}"
            centre(hdr, 1)
            divider(9)

            counters = [
                ("DRAWN", str(self._draws_done),
                 on_col, glow_col),
                ("LEFT",  str(self._draws_total - self._draws_done),
                 on_col, glow_col),
                ("HITS",  str(self._hits),
                 hi_on if self._hits > 0 else on_col, hi_glow),
                ("NEED",  str(max(0, self._spots - self._hits)),
                 hi_on if 0 < self._spots - self._hits <= 2 else on_col, hi_glow),
            ]
            ctr_start = 11
            ctr_end   = rows // 2 - 2
            ctr_step  = max(8, (ctr_end - ctr_start) // 4)
            for i, (lbl, val, vc, vg) in enumerate(counters):
                dr = ctr_start + i * ctr_step
                txt(lbl, 2, dr, 1, 1, dim_on, dim_glow)
                right(val, dr, 1, 1, vc, vg)

            div_row = ctr_start + 4 * ctr_step
            divider(div_row)
            txt("CATCH", 2, div_row + 2, 1, 1, dim_on, dim_glow)
            right("PRIZE", div_row + 2, 1, 1, dim_on, dim_glow)
            draw_prize_table(self._prize_rows(), div_row + 10, rows - 2)

        # ──────────────────────────────────────────────────────────────────
        else:
            if self._spots == 0:
                centre("PICK NUMBERS",   rows // 2 - 6, oc=on_col)
                divider(rows // 2 + 2)
                centre("TO SEE PAYOUTS", rows // 2 + 4, oc=dim_on, gc=dim_glow)
            else:
                hdr = f"PICK {self._spots}  BET {self._bet}"
                if self._prize_mult > 1:
                    hdr += f"  X{self._prize_mult}"
                centre(hdr, 1)
                divider(9)
                txt("CATCH", 2, 11, 1, 1, dim_on, dim_glow)
                right("PRIZE", 11, 1, 1, dim_on, dim_glow)
                divider(19)
                draw_prize_table(self._prize_rows(), 21, rows - 10)
                divider(rows - 8)
                txt("CREDITS", 2, rows - 6, 1, 1, dim_on, dim_glow)
                right(format_credits(self._balance), rows - 6, 1, 1, on_col, glow_col)

        p.end()




class DoubleUpOverlay(QWidget):
    """In-cabinet double-up prompt that replaces the modal QMessageBox."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self._on_yes = None
        self._on_no = None
        self._timeout = QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._auto_decline)
        self._seconds_left = 10

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        self.title = QLabel("★  DOUBLE UP?  ★")
        self.title.setFont(QFont("Arial Black", 18, QFont.Weight.ExtraBold))
        self.title.setStyleSheet("color: #ffdd33; background: transparent;")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.detail = QLabel("")
        self.detail.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.detail.setStyleSheet("color: #aaccee; background: transparent;")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

        self.timer_lbl = QLabel("")
        self.timer_lbl.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        self.timer_lbl.setStyleSheet("color: #6688aa; background: transparent;")
        self.timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.timer_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)
        self.yes_btn = CabinetButton("YES — 3× WINS", "#00cc55", "#000000")
        self.yes_btn.setFixedHeight(44)
        self.yes_btn.clicked.connect(self._choose_yes)
        btn_row.addWidget(self.yes_btn)
        self.no_btn = CabinetButton("NO — CONTINUE", "#4499dd", "#ffffff")
        self.no_btn.setFixedHeight(44)
        self.no_btn.clicked.connect(self._choose_no)
        btn_row.addWidget(self.no_btn)
        layout.addLayout(btn_row)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._countdown)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(4, 12, 24, 210))
        margin = 18
        panel = self.rect().adjusted(margin, margin, -margin, -margin)
        grad = QLinearGradient(0, panel.top(), 0, panel.bottom())
        grad.setColorAt(0.0, QColor("#1a2a40"))
        grad.setColorAt(1.0, QColor("#0a1420"))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QColor("#ffcc00"), 3))
        p.drawRoundedRect(panel, 10, 10)
        p.end()

    def show_prompt(self, bet_amount: int, on_yes, on_no, timeout_s: int = 10):
        self._on_yes = on_yes
        self._on_no = on_no
        self._seconds_left = timeout_s
        self.detail.setText(
            f"Pay another {format_credits(bet_amount)} credits now.\n"
            f"All wins (including jackpot) pay 3× for the rest of this draw."
        )
        self.timer_lbl.setText(f"Auto-declines in {self._seconds_left}s")
        self.setVisible(True)
        self.raise_()
        self._tick.start(1000)
        self._timeout.start(timeout_s * 1000)

    def _countdown(self):
        self._seconds_left = max(0, self._seconds_left - 1)
        self.timer_lbl.setText(f"Auto-declines in {self._seconds_left}s")

    def _choose_yes(self):
        self._close()
        if self._on_yes:
            self._on_yes()

    def _choose_no(self):
        self._close()
        if self._on_no:
            self._on_no()

    def _auto_decline(self):
        self._close()
        if self._on_no:
            self._on_no()

    def _close(self):
        self._tick.stop()
        self._timeout.stop()
        self.setVisible(False)



class CabinetButton(QPushButton):
    def __init__(self, text, accent="#ffcc00", fg="#000000", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Arial Black", 11, QFont.Weight.ExtraBold))
        self.setFixedHeight(48)
        dark = "#886600" if accent == "#ffcc00" else "#224466"
        hover = "#ffee66" if accent == "#ffcc00" else "#66bbff"
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {accent}, stop:1 {dark});
                color: {fg}; border: 2px solid {"#ffee88" if accent == "#ffcc00" else "#66bbff"};
                border-radius: 6px; padding: 0 18px; letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {hover}, stop:1 {accent});
            }}
            QPushButton:pressed {{ background: {dark}; }}
            QPushButton:disabled {{ background:#1a2a3a; color:#334455; border-color:#223344; }}
        """)


class KenoGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Super Double Up — Video Keno")
        self.setMinimumSize(1020, 860)

        self.balance = _load_balance()
        self.bet = 1
        self.player_picks: set[int] = set()
        self.drawn_numbers: list[int] = []
        self.draw_index = 0
        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self._reveal_next)
        self.win_count_timer = QTimer()
        self._jackpot_tick_count = 0
        self.jackpot_drip_timer = QTimer(self)
        self.jackpot_drip_timer.setInterval(JACKPOT_TICK_MS)
        self.jackpot_drip_timer.timeout.connect(self._drip_jackpots)
        self.win_count_timer.timeout.connect(self._animate_winnings_tick)
        self.balls: dict[int, BallWidget] = {}
        self.max_picks = 10
        self._current_hits = 0
        self._win_count_target = 0
        self._win_count_value = 0
        self._win_count_step = 0
        self._win_flash_on = False
        self._double_up_prompted = False
        self._double_up_active = False
        self._bet_before_double_up = self.bet
        self._chip_animations: list[QPropertyAnimation] = []
        self._sfx = SoundEngine()
        saved = _load_jackpots()
        self._jackpots: dict[int, int] = {
            t: saved.get(t, s) for t, s in JACKPOT_SEEDS.items()
        }        # Session stats (reset each application launch)
        self._sess_rounds: int = 0
        self._sess_total:  int = 0
        self._sess_best:   int = 0
        self._broke = self.balance <= 0

        self._build_ui()
        self._refresh_info()
        if self._broke:
            self._enter_broke_state(play_sound=False)
        self.jackpot_drip_timer.start()

    # ─── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        bg = GradientPanel("#0d1f3c", "#060e1a")
        self.setCentralWidget(bg)
        outer = QVBoxLayout(bg)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = CabinetHeader()
        header.setFixedHeight(88)
        outer.addWidget(header)

        # Middle: board + payout panel side by side
        mid = QWidget()
        mid.setStyleSheet("background: transparent;")
        mh = QHBoxLayout(mid)
        mh.setContentsMargins(18, 14, 18, 8)
        mh.setSpacing(14)

        left = QWidget()
        left.setStyleSheet("background: transparent;")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(10)
        self._board_host = QWidget()
        board_layout = QVBoxLayout(self._board_host)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(0)
        board_layout.addWidget(self._make_board())
        lv.addWidget(self._board_host, 1)
        lv.addWidget(self._make_drawn_row())

        self.double_up_overlay = DoubleUpOverlay(self._board_host)
        self.double_up_overlay.setGeometry(self._board_host.rect())
        self._board_host.installEventFilter(self)

        mh.addWidget(left, 1)

        right_side = QWidget()
        right_side.setStyleSheet("background: transparent;")
        rv = QVBoxLayout(right_side)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)

        self.jackpot_strip = JackpotStrip()
        rv.addWidget(self.jackpot_strip, 0)

        self.payout_dmd = PayoutDMD()
        self.game_state_panel = GameStatePanel()
        rv.addWidget(self.game_state_panel, 0)
        rv.addWidget(self.payout_dmd, 1)

        self.random_pick_btn = CabinetButton("RANDOM 10", "#4499dd", "#ffffff")
        self.random_pick_btn.setFixedWidth(185)
        self.random_pick_btn.setFixedHeight(44)
        self.random_pick_btn.clicked.connect(self._random_pick_10)
        rv.addWidget(self.random_pick_btn, 0)

        mh.addWidget(right_side, 0)

        outer.addWidget(mid, 1)

        # Bottom cabinet panel
        bottom = BottomPanel()
        bottom.setFixedHeight(120)
        bh = QHBoxLayout(bottom)
        bh.setContentsMargins(20, 10, 20, 10)
        bh.setSpacing(16)
        bh.addLayout(self._make_credit_block())
        bh.addStretch()
        bh.addLayout(self._make_bet_block())
        bh.addStretch()
        bh.addLayout(self._make_action_block())
        outer.addWidget(bottom)

        # Status ticker
        self.status_lbl = QLabel("● SELECT YOUR NUMBERS AND PRESS PLAY")
        self.status_lbl.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        self.status_lbl.setStyleSheet(
            "color: #44aaff; background: #040c18; padding: 5px 14px;"
            "border-top: 1px solid #1a3a6a;"
        )
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.status_lbl)

    def _make_board(self):
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame { background: #040d1a; border: 2px solid #1a3a6a; border-radius: 8px; }
        """)
        grid = QGridLayout(frame)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(14)
        grid.setContentsMargins(14, 14, 14, 14)
        for n in range(1, 81):
            ball = BallWidget(n)
            ball.set_click_callback(self._toggle_pick)
            self.balls[n] = ball
            row, col = divmod(n - 1, 10)
            grid.addWidget(ball, row, col)
        return frame

    def _make_drawn_row(self):
        wrap = QWidget()
        wrap.setStyleSheet("background: transparent;")
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        hdr = QLabel("DRAWN  NUMBERS")
        hdr.setFont(QFont("Courier New", 8, QFont.Weight.Bold))
        hdr.setStyleSheet(
            "color: #336688; letter-spacing: 5px; background: transparent;"
        )
        v.addWidget(hdr)

        # One fixed-size row with exactly DRAWN_CHIPS_COUNT slots, no scrolling.
        chip_row = QWidget()
        chip_row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(chip_row)
        row_layout.setSpacing(DRAWN_CHIP_SPACING)
        row_layout.setContentsMargins(0, 0, 0, 0)

        slot_size = DRAWN_CHIP_SIZE + 2  # a little breathing room
        self._drawn_slots: list[QWidget] = []
        for _ in range(DRAWN_CHIPS_COUNT):
            slot = QWidget(chip_row)
            slot.setFixedSize(slot_size, slot_size)
            slot.setStyleSheet("background: transparent;")
            row_layout.addWidget(slot)
            self._drawn_slots.append(slot)

        chip_row.setFixedHeight(slot_size)
        v.addWidget(chip_row)
        return wrap

    def _make_credit_block(self):
        v = QVBoxLayout()
        v.setSpacing(4)

        label_row = QHBoxLayout()
        label_row.setSpacing(8)

        lbl = QLabel("CREDITS")
        lbl.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        lbl.setStyleSheet(
            "color: #336688; letter-spacing: 3px; background:transparent;"
        )
        label_row.addWidget(lbl)

        win_lbl = QLabel("WINNINGS")
        win_lbl.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        win_lbl.setStyleSheet(
            "color: #336688; letter-spacing: 3px; background:transparent;"
        )
        label_row.addWidget(win_lbl)
        v.addLayout(label_row)

        display_row = QHBoxLayout()
        display_row.setSpacing(8)

        self.credit_display = DigitalLabel(format_credits(1000))
        self.credit_display.setFixedWidth(160)
        display_row.addWidget(self.credit_display)

        self.winnings_display = DigitalLabel(format_winnings(0))
        self.winnings_display.setFixedWidth(150)
        self.winnings_display.setStyleSheet("""
            color: #336688; background: #000a14;
            border: 2px solid #1a4a7a; border-radius: 4px; padding: 4px 12px;
        """)
        display_row.addWidget(self.winnings_display)

        v.addLayout(display_row)
        return v

    def _make_bet_block(self):
        v = QVBoxLayout()
        v.setSpacing(4)
        lbl = QLabel("BET PER GAME")
        lbl.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        lbl.setStyleSheet(
            "color: #336688; letter-spacing: 3px; background:transparent;"
        )
        v.addWidget(lbl)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.bet_minus = self._small_btn("◀", self._decrease_bet)
        row.addWidget(self.bet_minus)
        self.bet_display = DigitalLabel(format_credits(1))
        self.bet_display.setFixedWidth(80)
        row.addWidget(self.bet_display)
        self.bet_plus = self._small_btn("▶", self._increase_bet)
        row.addWidget(self.bet_plus)
        v.addLayout(row)
        self.picks_display = QLabel("PICKS: 0 / 10")
        self.picks_display.setFont(QFont("Courier New", 9, QFont.Weight.Bold))
        self.picks_display.setStyleSheet("color: #88aacc; background: transparent;")
        v.addWidget(self.picks_display)
        return v

    def _make_action_block(self):
        v = QVBoxLayout()
        v.setSpacing(8)
        self.play_btn = CabinetButton("▶  PLAY", "#ffcc00", "#000000")
        self.play_btn.setFixedWidth(150)
        self.play_btn.clicked.connect(self._start_draw)
        v.addWidget(self.play_btn)
        self.clear_btn = CabinetButton("CLEAR", "#4499dd", "#ffffff")
        self.clear_btn.setFixedWidth(150)
        self.clear_btn.clicked.connect(self._clear_picks)
        v.addWidget(self.clear_btn)
        self.rebuy_btn = CabinetButton("REBUY 1K", "#ff6644", "#000000")
        self.rebuy_btn.setFixedWidth(150)
        self.rebuy_btn.clicked.connect(self._rebuy_credits)
        self.rebuy_btn.setVisible(False)
        v.addWidget(self.rebuy_btn)
        return v

    def _small_btn(self, text, slot):
        btn = QPushButton(text)
        btn.setFixedSize(32, 32)
        btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        btn.setStyleSheet("""
            QPushButton { background:#0d2040; color:#ffcc00; border:2px solid #1a4a7a; border-radius:4px; }
            QPushButton:hover { background:#1a3a6a; }
            QPushButton:pressed { background:#091528; }
        """)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(slot)
        return btn

    # ─── Logic ────────────────────────────────────────────────────────────────

    def _refresh_info(self):
        self.credit_display.setText(format_credits(self.balance))
        self.bet_display.setText(format_credits(self.bet))
        n = len(self.player_picks)
        self.picks_display.setText(f"PICKS: {n} / {self.max_picks}")
        self.picks_display.setStyleSheet(
            f"color: {'#ffcc00' if n > 0 else '#88aacc'}; background: transparent;"
        )
        prize_multiplier = 3 if self._double_up_active else 1
        jp_val = jackpot_value_for_bet(self._jackpots, n)
        self.payout_dmd.update_spots(n, prize_multiplier=prize_multiplier,
                                     jackpot_value=jp_val, bet=self.bet)
        self.payout_dmd.set_balance(self.balance)
        self.game_state_panel.set_balance(self.balance)
        self.jackpot_strip.set_jackpot(n, jp_val)
        if self.bet > self.balance and self.balance > 0:
            self.bet = max(1, self.balance)
            self.bet_display.setText(format_credits(self.bet))

    def _toggle_pick(self, number: int):
        if self.draw_timer.isActive():
            return
        if number in self.player_picks:
            self.player_picks.remove(number)
            self.balls[number].set_state("default")
        elif len(self.player_picks) < self.max_picks:
            self.player_picks.add(number)
            self.balls[number].set_state("selected")
        self._refresh_info()

    def _random_pick_10(self):
        """Clear current picks and randomly select exactly 10 numbers."""
        if self.draw_timer.isActive():
            return

        self.player_picks.clear()
        self._reset_board_display()

        self.player_picks = set(random.sample(range(1, 81), self.max_picks))
        for n in self.player_picks:
            self.balls[n].set_state("selected")

        self._reset_winnings_indicator()
        self._refresh_info()
        self._set_status("● RANDOM 10 NUMBERS SELECTED — PRESS PLAY", "#44aaff")

    def _clear_picks(self):
        if self.draw_timer.isActive():
            return
        for n in list(self.player_picks):
            self.balls[n].set_state("default")
        self.player_picks.clear()
        self._reset_board_display()
        self._refresh_info()
        self._set_status("● SELECT YOUR NUMBERS AND PRESS PLAY", "#44aaff")

    def _reset_board_display(self):
        for n, ball in self.balls.items():
            if n not in self.player_picks:
                ball.set_state("default")

        for anim in self._chip_animations:
            anim.stop()
        self._chip_animations.clear()

        for slot in self._drawn_slots:
            for child in slot.findChildren(RollingDrawnBall):
                child.deleteLater()

    def _increase_bet(self):
        steps = [1, 2, 5, 10, 25, 50, 100]
        idx = next((i for i, v in enumerate(steps) if v > self.bet), len(steps) - 1)
        self.bet = min(steps[idx], self.balance)
        self._refresh_info()

    def _decrease_bet(self):
        steps = [1, 2, 5, 10, 25, 50, 100]
        idx = next((i for i, v in enumerate(steps) if v >= self.bet), 0) - 1
        self.bet = steps[max(idx, 0)]
        self._refresh_info()

    def _start_draw(self):
        if self._broke or self.balance <= 0:
            self._enter_broke_state(play_sound=False)
            return
        if not self.player_picks:
            self._set_status("⚠  SELECT AT LEAST 1 NUMBER!", "#ff6633")
            return
        if self.bet > self.balance:
            self._set_status("⚠  INSUFFICIENT CREDITS!", "#ff6633")
            return

        self.balance -= self.bet
        spots = len(self.player_picks)
        total_contrib = max(1, int(self.bet * JACKPOT_CONTRIB))
        primary = max(1, int(total_contrib * JACKPOT_PRIMARY_SHARE))
        shared = total_contrib - primary
        self._jackpots[spots] = self._jackpots.get(spots, JACKPOT_SEEDS[spots]) + primary
        other_tiers = [t for t in self._jackpots if t != spots]
        if other_tiers and shared > 0:
            each = max(1, shared // len(other_tiers))
            for t in other_tiers:
                self._jackpots[t] += each
        _save_state(self._jackpots, self.balance)
        self._current_hits = 0
        self._double_up_prompted = False
        self._double_up_active = False
        self._bet_before_double_up = self.bet
        self._reset_winnings_indicator()
        self._refresh_info()
        self.jackpot_strip.set_jackpot(
            spots, jackpot_value_for_bet(self._jackpots, spots), flash=True
        )
        self.play_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.random_pick_btn.setEnabled(False)

        self._reset_board_display()
        for n in self.player_picks:
            self.balls[n].set_state("selected")

        self.drawn_numbers = random.sample(range(1, 81), 20)
        self.draw_index = 0
        self._set_status(
            f"● DRAWING…  +{format_credits(total_contrib)} → {spots}-SPOT JACKPOT",
            "#ffcc00",
        )
        jp_val = jackpot_value_for_bet(self._jackpots, spots)
        self.payout_dmd.start_draw(spots, self.bet, 1, jp_val, self.balance)
        self.game_state_panel.start_draw(spots)
        self.draw_timer.start(DRAW_INTERVAL_MS)

    def _reveal_next(self):
        if self.draw_index >= len(self.drawn_numbers):
            self.draw_timer.stop()
            self._finish_round()
            return

        n = self.drawn_numbers[self.draw_index]
        self.draw_index += 1
        is_hit = n in self.player_picks
        self.balls[n].set_state("hit" if is_hit else "drawn_miss")

        if is_hit:
            self._current_hits += 1

        # Update dot tracker and draw counters on every ball
        self.game_state_panel.register_hit(is_hit)
        self.payout_dmd.update_draw(self._current_hits, self.draw_index)

        self._add_drawn_chip(n, is_hit)
        self._sfx.play("hit" if is_hit else "tick")

        halfway_index = len(self.drawn_numbers) // 2
        if (
            self.draw_index == halfway_index
            and self._current_hits >= 3
            and not self._double_up_prompted
        ):
            self._double_up_prompted = True
            self.draw_timer.stop()
            QTimer.singleShot(0, self._offer_double_up)

    def _add_drawn_chip(self, number: int, is_hit: bool):
        """Drop a drawn-number ball down from above into its pre-allocated slot."""
        slot_idx = self.draw_index - 1  # draw_index was already incremented
        if slot_idx >= len(self._drawn_slots):
            return
        slot = self._drawn_slots[slot_idx]
        slot_size = slot.width()
        offset = (slot_size - DRAWN_CHIP_SIZE) // 2

        chip = RollingDrawnBall(number, is_hit, slot)
        chip.move(offset, -DRAWN_CHIP_SIZE)
        chip.show()

        drop_anim = QPropertyAnimation(chip, b"pos", self)
        drop_anim.setDuration(DRAWN_CHIP_SLIDE_MS)
        drop_anim.setStartValue(QPoint(offset, -DRAWN_CHIP_SIZE))
        drop_anim.setEndValue(QPoint(offset, offset))
        drop_anim.setEasingCurve(QEasingCurve.Type.OutBounce)

        spin_anim = QPropertyAnimation(chip, b"angle", self)
        spin_anim.setDuration(DRAWN_CHIP_SLIDE_MS // 2)
        spin_anim.setStartValue(0.0)
        spin_anim.setEndValue(-180.0)
        spin_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(drop_anim)
        group.addAnimation(spin_anim)

        def cleanup():
            chip.move(offset, offset)
            chip.set_angle(0.0)
            if group in self._chip_animations:
                self._chip_animations.remove(group)

        group.finished.connect(cleanup)
        self._chip_animations.append(group)
        group.start()

    def _offer_double_up(self):
        """Pause halfway through the draw and offer a higher-risk double-up."""
        if self.balance < self.bet:
            self._set_status(
                "●  DOUBLE UP AVAILABLE — INSUFFICIENT CREDITS TO DOUBLE BET",
                "#ffaa00",
            )
            self.draw_timer.start(DRAW_INTERVAL_MS)
            return

        self._set_status(
            "★  DOUBLE UP?  DOUBLE YOUR BET AND WINS PAY 3×  ★",
            "#ffcc00",
        )
        self.double_up_overlay.setGeometry(self._board_host.rect())
        self.double_up_overlay.show_prompt(
            self.bet,
            on_yes=self._accept_double_up,
            on_no=self._decline_double_up,
            timeout_s=10,
        )

    def _accept_double_up(self):
        self._bet_before_double_up = self.bet
        self.balance -= self.bet
        self.bet *= 2
        self._double_up_active = True
        self.credit_display.setText(format_credits(self.balance))
        self.bet_display.setText(format_credits(self.bet))
        spots = len(self.player_picks)
        jp_val = jackpot_value_for_bet(self._jackpots, spots)
        self.payout_dmd.start_draw(spots, self.bet, 3, jp_val)
        self.payout_dmd.update_draw(self._current_hits)
        self._set_status(
            "★  DOUBLE UP ACTIVE — PAYOUT TABLE NOW SHOWS 3× WINS  ★",
            "#00ff66",
        )
        self.draw_timer.start(DRAW_INTERVAL_MS)

    def _decline_double_up(self):
        self._set_status(
            "●  DOUBLE UP DECLINED — CONTINUING DRAW",
            "#44aaff",
        )
        self.draw_timer.start(DRAW_INTERVAL_MS)

    def _finish_round(self):
        drawn_set = set(self.drawn_numbers)
        hits = len(self.player_picks & drawn_set)
        spots = len(self.player_picks)

        pass  # jackpot panel removed
        self.game_state_panel.stop_draw()
        final_table_bet        = self.bet
        final_table_multiplier = 3 if self._double_up_active else 1
        final_table_jackpot    = jackpot_value_for_bet(self._jackpots, spots)

        # Jackpot: full catch on any pick count wins the progressive pot.
        # Hard rule: jackpot never pays less than the best fixed prize for this
        # ticket (full-catch mult if present, otherwise the next-lower catch),
        # scaled by bet and double-up. Seeds are sized for max bet, but the
        # floor still protects mid-growth / high-bet edge cases.
        jackpot_win    = (hits == spots) and (spots in self._jackpots)
        jackpot_amount = 0
        win_multiplier = 3 if self._double_up_active else 1
        if jackpot_win:
            table = PAYOUTS.get(spots, {})
            # Fixed full-catch mult (0 on progressive-only rows 7-10)
            full_mult = table.get(spots, 0)
            if full_mult > 0:
                floor_mult = full_mult
            else:
                # Best non-full catch on this ticket
                floor_mult = max((m for h, m in table.items() if h < spots), default=0)
            floor = floor_mult * self.bet * win_multiplier
            # Double-up multiplies the pot as advertised in the overlay
            pot_payout = final_table_jackpot * win_multiplier
            jackpot_amount = max(pot_payout, floor)
            self._jackpots[spots] = JACKPOT_SEEDS[spots]   # reset to seed
            _save_state(self._jackpots, self.balance)
            final_table_jackpot = JACKPOT_SEEDS[spots]

        # Fixed-pay tiers: multiplier × bet (jackpot catch row is 0 in table)
        payout_mult = PAYOUTS.get(spots, {}).get(hits, 0)
        if jackpot_win:
            payout_mult = 0
        winnings = payout_mult * self.bet * win_multiplier + jackpot_amount
        self.balance += winnings

        if self._double_up_active:
            self.bet = self._bet_before_double_up

        self.credit_display.setText(format_credits(self.balance))
        self.bet_display.setText(format_credits(self.bet))
        self.picks_display.setText(f"PICKS: {spots} / {self.max_picks}")
        self.picks_display.setStyleSheet(
            f"color: {'#ffcc00' if spots > 0 else '#88aacc'}; background: transparent;"
        )

        # Show WIN/NO WIN on the game state panel
        self.game_state_panel.show_result(winnings)

        self._start_winnings_count(winnings)

        # Update session stats
        self._sess_rounds += 1
        self._sess_total  += winnings
        self._sess_best    = max(self._sess_best, winnings)

        # Stop draw mode on payout DMD and show result + winner highlight
        self.payout_dmd.stop_draw()
        winner_hits = hits if (payout_mult > 0 or jackpot_win) else -1
        self.payout_dmd.update_spots(
            spots, hits=hits, winner_hits=winner_hits,
            prize_multiplier=final_table_multiplier,
            jackpot_value=final_table_jackpot, bet=final_table_bet,
        )
        # Show session stats for 3.5 s after the round
        self.payout_dmd.show_result(self._sess_rounds, self._sess_total, self._sess_best,
                                    hits, spots, winnings)


        if jackpot_win:
            msg = f"★★  JACKPOT!  CATCH {hits}  —  {format_credits(jackpot_amount)}  ★★"
            color = "#ffdd33"
            self._sfx.play("jackpot_fanfare")
            QTimer.singleShot(400, lambda: QMessageBox.information(
                self, "★  JACKPOT!  ★",
                f"CATCH {hits} OF {spots}\n\nYOU HIT THE PROGRESSIVE JACKPOT!\n\n"
                f"  {format_credits(jackpot_amount)} CREDITS  \n\n"
                f"Jackpot resets to {format_credits(JACKPOT_SEEDS[spots])}.",
            ))
        elif winnings > 0:
            if win_multiplier > 1:
                msg = f"★  {hits} OF {spots} — DOUBLE UP WIN  {format_credits(winnings)}  (3×)  ★"
            else:
                msg = f"★  {hits} OF {spots} — WIN  {format_credits(winnings)}  ★"
            color = "#00ff66"
        elif hits > 0:
            msg = f"●  {hits} OF {spots} MATCHED — NO PAYOUT"
            color = "#ffaa00"
        else:
            msg = "✗  NO MATCHES — BETTER LUCK NEXT TIME"
            color = "#ff4444"

        self._set_status(msg, color)
        self.play_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.random_pick_btn.setEnabled(True)
        self.jackpot_strip.set_jackpot(
            spots, jackpot_value_for_bet(self._jackpots, spots)
        )

        if self.balance <= 0:
            self._enter_broke_state(play_sound=True)

    def _reset_winnings_indicator(self):
        self.win_count_timer.stop()
        self._win_count_target = 0
        self._win_count_value = 0
        self._win_count_step = 0
        self._win_flash_on = False
        self.winnings_display.setText(format_winnings(0))
        self.winnings_display.setStyleSheet("""
            color: #336688; background: #000a14;
            border: 2px solid #1a4a7a; border-radius: 4px; padding: 4px 12px;
        """)

    def _start_winnings_count(self, winnings: int):
        self.win_count_timer.stop()
        self._win_count_target = max(0, winnings)
        self._win_count_value = 0
        self._win_flash_on = False

        if winnings <= 0:
            self.winnings_display.setText(format_winnings(0))
            self.winnings_display.setStyleSheet("""
                color: #334455; background: #000a14;
                border: 2px solid #1a4a7a; border-radius: 4px; padding: 4px 12px;
            """)
            return

        # About 24 ticks gives a fast slot-machine count-up without blocking the UI.
        self._win_count_step = max(1, winnings // 24)
        self.winnings_display.setText(format_winnings(0))
        self.winnings_display.setStyleSheet("""
            color: #00ff66; background: #031a0a;
            border: 2px solid #00ff66; border-radius: 4px; padding: 4px 12px;
        """)
        self.win_count_timer.start(35)

    def _animate_winnings_tick(self):
        self._win_count_value = min(
            self._win_count_target, self._win_count_value + self._win_count_step
        )
        self.winnings_display.setText(format_winnings(self._win_count_value))
        self._sfx.play("coin")

        if self._win_flash_on:
            bg, border = "#031a0a", "#00ff66"
        else:
            bg, border = "#173a00", "#ffee44"
        self._win_flash_on = not self._win_flash_on
        self.winnings_display.setStyleSheet(f"""
            color: #ffffff; background: {bg};
            border: 2px solid {border}; border-radius: 4px; padding: 4px 12px;
        """)

        if self._win_count_value >= self._win_count_target:
            self.win_count_timer.stop()
            self.winnings_display.setText(format_winnings(self._win_count_target))
            self.winnings_display.setStyleSheet("""
                color: #00ff66; background: #031a0a;
                border: 2px solid #ffee44; border-radius: 4px; padding: 4px 12px;
            """)
            self._sfx.play("win_fanfare")

    def _set_status(self, msg, color):
        self.status_lbl.setText(msg)
        self.status_lbl.setStyleSheet(
            f"color: {color}; background: #040c18; padding: 5px 14px;"
            "border-top: 1px solid #1a3a6a; font-weight: bold;"
        )

    def _drip_jackpots(self):
        """Time-based progressive growth: +1 to every tier on a steady cadence.

        Runs while the cabinet is open so the jackpot strip always feels alive,
        not only when someone is betting. Bet contributions still apply on top.
        """
        for tier in self._jackpots:
            self._jackpots[tier] = self._jackpots.get(tier, JACKPOT_SEEDS[tier]) + JACKPOT_TICK_AMOUNT
        self._jackpot_tick_count += 1
        if self._jackpot_tick_count % JACKPOT_TICK_SAVE_EVERY == 0:
            _save_state(self._jackpots, self.balance)
        # Refresh the visible strip for the current pick count (odometer tick)
        spots = len(self.player_picks)
        if spots > 0 and hasattr(self, "jackpot_strip"):
            val = jackpot_value_for_bet(self._jackpots, spots)
            self.jackpot_strip.set_jackpot(spots, val, tick=True)
            # Keep payout DMD jackpot figure in sync when idle
            if not self.draw_timer.isActive() and hasattr(self, "payout_dmd"):
                try:
                    self.payout_dmd.update_spots(
                        spots,
                        jackpot_value=val,
                        bet=self.bet,
                    )
                except TypeError:
                    pass

    def _enter_broke_state(self, play_sound: bool = True):
        """Player is out of credits — disable play and offer a rebuy."""
        self._broke = True
        self.balance = 0
        self.play_btn.setEnabled(False)
        self.play_btn.setVisible(False)
        self.rebuy_btn.setVisible(True)
        self.rebuy_btn.setEnabled(True)
        if play_sound:
            self._sfx.play("broke")
        self._set_status(
            "✗  OUT OF CREDITS — PRESS REBUY TO CONTINUE",
            "#ff6644",
        )
        self._refresh_info()
        _save_state(self._jackpots, self.balance)

    def _rebuy_credits(self):
        """Grant a fresh stack of credits after going broke."""
        self.balance = DEFAULT_BALANCE
        self._broke = False
        self.rebuy_btn.setVisible(False)
        self.play_btn.setVisible(True)
        self.play_btn.setEnabled(True)
        self.bet = min(self.bet, self.balance)
        self._refresh_info()
        self._set_status(
            f"●  REBUY COMPLETE — {format_credits(DEFAULT_BALANCE)} CREDITS LOADED",
            "#00ff66",
        )
        _save_state(self._jackpots, self.balance)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "_board_host", None) and event.type() == event.Type.Resize:
            self.double_up_overlay.setGeometry(self._board_host.rect())
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        _save_state(self._jackpots, self.balance)
        self._sfx.cleanup()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#060e1a"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#aaccee"))
    app.setPalette(palette)
    win = KenoGame()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

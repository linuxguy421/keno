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
# Values are multipliers applied to the bet (e.g. 3 → win 3× your bet).
# Jackpot-tier full catches use the progressive jackpot instead of a multiplier.
# The jackpot display/payoff is scaled by bet so lesser catches cannot exceed it.
PAYOUTS = {
    1:  {1: 3},
    2:  {2: 12,    1: 1},
    3:  {3: 45,    2: 5,    1: 1},
    4:  {4: 120,   3: 10,   2: 2},
    5:  {5: 450,   4: 20,   3: 3,   2: 1},
    6:  {6: 1500,  5: 50,   4: 8,   3: 2},
    7:  {7: 5000,  6: 100,  5: 15,  4: 3,  3: 1},
    8:  {8: 0,     7: 500,  6: 50,  5: 12, 4: 2},         # 8-of-8 → jackpot
    9:  {9: 0,     8: 2000, 7: 100, 6: 20, 5: 5,  4: 1},  # 9-of-9 → jackpot
    10: {10: 0,    9: 5000, 8: 500, 7: 50, 6: 10, 5: 2, 4: 1, 3: 1},  # 10 → jackpot
}

BASE_BET = 1   # reference bet the jackpot seeds are calibrated to


# ─── Progressive jackpot seeds & contribution rate ────────────────────────────
# Seeds are calibrated to BASE_BET=1. At higher bets the seed scales linearly.
JACKPOT_SEEDS   = {8: 5_000,  9: 15_000,  10: 50_000}
JACKPOT_CONTRIB = 0.02   # fraction of every bet added to each pot


def jackpot_seed_for_bet(tier: int, bet: int) -> int:
    """Return the reset seed for `tier` scaled to the current bet."""
    return JACKPOT_SEEDS[tier] * max(1, bet)


def jackpot_value_for_bet(pots: dict[int, int], tier: int, bet: int) -> int:
    """Return the displayed/payable jackpot for a tier at the current bet.

    Jackpot pots are stored as base credit amounts, while normal payouts scale
    by bet. Scaling the jackpot floor by bet keeps the jackpot above every
    lesser catch payout, so a near-jackpot result can never pay more than the
    jackpot itself.
    """
    if tier not in JACKPOT_SEEDS:
        return 0
    return max(int(pots.get(tier, 0)), jackpot_seed_for_bet(tier, bet))

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
        return max(v, DEFAULT_BALANCE)   # never load below the default
    except Exception:
        return DEFAULT_BALANCE


def _save_state(pots: dict[int, int], balance: int) -> None:
    try:
        existing = _load_save()
        existing["jackpots"] = {str(k): v for k, v in pots.items()}
        # Only persist balance if it's above the default (player earned it)
        if balance > DEFAULT_BALANCE:
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
    """Shared dot-matrix display renderer.  Subclasses use the protected helpers."""

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


class JackpotPanel(DmdPanel):
    """Pinball-machine DMD style progressive jackpot display.

    The panel is drawn like a classic amber/orange pinball dot-matrix display:
    a full field of dim inactive dots, with the jackpot tier and jackpot amount
    drawn by lighting dot clusters. This avoids font-mask and alpha-sampling
    problems and guarantees the jackpot amount is visible.
    """

    # Animation tick rates (ms) — inherited from DmdPanel

    def __init__(self, parent=None):
        super().__init__(parent=parent)  # DmdPanel handles idle timer + glyphs
        self.setFixedWidth(185)

        self._tier: int | None = None
        self._value: int = 0
        self._displayed_value: int = 0
        self._is_winner: bool = False

        # ── Odometer roll ────────────────────────────────────────────────
        self._digit_states: list[list] = []
        self._roll_timer = QTimer(self)
        self._roll_timer.timeout.connect(self._tick_roll)

        # ── Pulse / flicker ──────────────────────────────────────────────
        self._flicker_phase: float = 0.0
        self._pulsing: bool = False
        self._flicker_timer = QTimer(self)
        self._flicker_timer.timeout.connect(self._tick_flicker)

        # ── Draw animation (chase border + progress bar) ─────────────────
        self._draw_anim_active: bool = False
        self._draw_anim_frame:  int  = 0
        self._draw_anim_count:  int  = 0
        self._draw_anim_hits:   int  = 0
        self._draw_anim_total:  int  = DRAWN_CHIPS_COUNT
        self._chase_timer = QTimer(self)
        self._chase_timer.timeout.connect(self._tick_chase)

        # ── Winner cascade + afterglow ───────────────────────────────────
        self._cascade_step:    int   = -1
        self._afterglow_alpha: float = 0.0
        self._afterglow_dir:   int   = 0
        self._win_timer = QTimer(self)
        self._win_timer.timeout.connect(self._tick_win)

        # ── Win result display ───────────────────────────────────────────
        self._result_title:   str  = ""
        self._result_amount:  int  = 0
        self._showing_result: bool = False
        self._result_timer = QTimer(self)
        self._result_timer.setSingleShot(True)
        self._result_timer.timeout.connect(self._end_result_display)

        self.setFixedHeight(104)


    # ── Public API ─────────────────────────────────────────────────────────

    def set_active_tier(self, tier: int | None, value: int):
        self._tier = tier if (tier in JACKPOT_SEEDS) else None
        self._is_winner = False
        self._pulsing = False
        self._flicker_timer.stop()
        self._chase_timer.stop()
        self._draw_anim_active = False
        self._win_timer.stop()
        self._cascade_step = -1
        self._afterglow_alpha = 0.0
        self._showing_result = False
        self._result_timer.stop()
        self._set_value_instant(value)
        self.setVisible(self._tier is not None)
        self.update()

    def update_value(self, value: int):
        """Roll displayed digits to a new value."""
        if value == self._value:
            return
        self._value = value
        self._start_odometer_roll(value)

    def pulse(self):
        """Flicker mode: oscillate brightness when player is close to jackpot."""
        self._pulsing = True
        if not self._flicker_timer.isActive():
            self._flicker_timer.start(self._FLICKER_MS)

    def stop_pulse(self):
        self._pulsing = False
        self._flicker_phase = 0.0
        self._flicker_timer.stop()
        self._is_winner = False
        self.update()

    def start_draw_animation(self, total_draws: int = DRAWN_CHIPS_COUNT):
        if self._tier is None:
            return
        self._draw_anim_active = True
        self._draw_anim_frame = 0
        self._draw_anim_count = 0
        self._draw_anim_hits  = 0
        self._draw_anim_total = max(1, total_draws)
        if not self._chase_timer.isActive():
            self._chase_timer.start(self._CHASE_MS)
        self.update()

    def update_draw_animation(self, draw_count: int, hit_count: int):
        self._draw_anim_count = max(0, draw_count)
        self._draw_anim_hits  = max(0, hit_count)
        self.update()

    def stop_draw_animation(self):
        self._chase_timer.stop()
        self._draw_anim_active = False
        self._draw_anim_frame  = 0
        self.update()

    def show_result(self, hits: int, spots: int, winnings: int):
        """Briefly show catch result and winnings on the DMD, then revert to pot."""
        self._result_title  = f"CATCH {hits}/{spots}"
        self._result_amount = winnings
        self._showing_result = True
        # Roll the value row to the winnings amount
        if winnings > 0:
            self._start_odometer_roll(winnings)
        self._result_timer.start(3000)
        self.update()

    def _end_result_display(self):
        self._showing_result = False
        # Roll value row back to the jackpot pot
        self._start_odometer_roll(self._value)
        self.update()

    def show_winner(self):
        """Trigger cascade wipe then sustained afterglow."""
        self._flicker_timer.stop()
        self._pulsing = False
        self._is_winner = True
        self._cascade_step = 0
        self._afterglow_alpha = 0.0
        self._afterglow_dir   = 1
        if not self._win_timer.isActive():
            self._win_timer.start(self._AFTERGLOW_MS)
        self.update()

    # ── Tick handlers ──────────────────────────────────────────────────────

    def _tick_idle_base(self):
        """Override base: suppress repaint when other animations own the frame."""
        self._breathe_phase = (self._breathe_phase + 0.04) % (2 * math.pi)
        self._scanline_y    = (self._scanline_y    + 0.004) % 1.0
        if not self._is_winner and not self._pulsing:
            self.update()

    def _tick_flicker(self):
        """Advance flicker oscillation when pulsing."""
        self._flicker_phase = (self._flicker_phase + 0.18) % (2 * math.pi)
        self.update()

    def _tick_chase(self):
        """Advance border chase frame."""
        self._draw_anim_frame = (self._draw_anim_frame + 1) % 10000
        self.update()

    def _tick_roll(self):
        """Advance odometer digit roll — tick each digit toward its target."""
        DIGIT_ORDER = "0123456789"
        still_rolling = False
        for state in self._digit_states:
            disp, target, step = state
            if disp == target:
                continue
            still_rolling = True
            state[2] += 1
            if state[2] >= self._ROLL_STEPS or disp not in DIGIT_ORDER:
                state[0] = target
                state[2] = 0
            else:
                idx = (DIGIT_ORDER.index(disp) + 1) % len(DIGIT_ORDER)
                state[0] = DIGIT_ORDER[idx]
        self.update()
        if not still_rolling:
            self._roll_timer.stop()
            # Snap displayed value to final
            self._displayed_value = self._value

    def _tick_win(self):
        """Cascade wipe columns left→right then hold afterglow, then slow decay."""
        if self._cascade_step >= 0:
            self._cascade_step += 3          # advance wipe by 3 dot-columns/tick
            cols = 104
            if self._cascade_step > cols:
                self._cascade_step = -1      # wipe done; switch to glow hold
                self._afterglow_alpha = 1.0
                self._afterglow_dir   = 0    # hold
                # Start slow decay after 1.2 s
                QTimer.singleShot(1200, self._begin_afterglow_decay)
        self.update()

    def _begin_afterglow_decay(self):
        self._afterglow_dir = -1

    # ── Internal helpers ───────────────────────────────────────────────────

    def _set_value_instant(self, value: int):
        """Set value immediately with no roll animation."""
        self._value = value
        self._displayed_value = value
        text = format_credits(value)
        self._digit_states = [[ch, ch, 0] for ch in text]
        self._roll_timer.stop()

    def _start_odometer_roll(self, new_value: int):
        """Diff old vs new digit string and kick off roll for changed digits."""
        old_text = format_credits(self._displayed_value)
        new_text = format_credits(new_value)
        # Pad shorter string on the left with spaces
        diff = len(new_text) - len(old_text)
        if diff > 0:
            old_text = " " * diff + old_text
        elif diff < 0:
            new_text = " " * (-diff) + new_text
        self._digit_states = [
            [o, n, 0] for o, n in zip(old_text, new_text)
        ]
        if not self._roll_timer.isActive():
            self._roll_timer.start(self._ROLL_MS)

    def _current_display_text(self) -> str:
        return "".join(s[0] for s in self._digit_states) if self._digit_states else format_credits(self._value)

    def _brightness(self) -> float:
        if self._is_winner:
            if self._afterglow_dir == -1:
                self._afterglow_alpha = max(0.0, self._afterglow_alpha - 0.008)
            return max(0.85, self._afterglow_alpha)
        if self._pulsing:
            return 0.60 + 0.40 * (0.5 + 0.5 * math.sin(self._flicker_phase))
        return self._breathe()

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        bri  = self._brightness()

        border_override = None
        if self._is_winner:
            border_override = QColor(
                68  + int((255 - 68)  * self._afterglow_alpha * 0.9),
                36  + int((176 - 36) * self._afterglow_alpha * 0.9), 0)

        if self._is_winner:
            on_base = QColor("#ffd35a"); glow_base = QColor("#ff9d00")
        elif self._pulsing:
            on_base = QColor("#ffbf33"); glow_base = QColor("#ff8c00")
        else:
            on_base = QColor("#ff7a00"); glow_base = QColor("#b84a00")
        off_col  = QColor("#2b1202")
        on_col   = self._scale_bri(on_base,   bri)
        glow_col = self._scale_bri(glow_base, bri)

        scanline = not self._is_winner and not self._pulsing
        pitch, dot, gx0, gy0 = self._paint_base(
            p, W, H, off_col, on_col, border_override=border_override,
            scanline=scanline)
        cols = self._dmd_cols
        rows = self._dmd_rows

        # ── Title row ─────────────────────────────────────────────────────
        if self._showing_result:
            title = self._result_title
        elif self._draw_anim_active:
            drawn  = self._draw_anim_count; hits = self._draw_anim_hits
            spots  = self._tier or 0;       rem  = self._draw_anim_total - drawn
            needed = spots - hits
            if spots > 0 and needed > 0 and needed <= rem <= needed + 2:
                title = f"NEED {needed} MORE"
            else:
                title = f"{drawn:02d} DRAWN  {hits} HIT"
        else:
            title = f"CATCH {self._tier} JACKPOT" if self._tier else "JACKPOT"

        tw = self._text_width(title, 1, 1)
        self._draw_dmd_text(p, title, gx0, gy0, pitch,
                            max(0, (cols - tw) // 2), 3, 1, 1, dot, on_col, glow_col)

        # ── Value row ─────────────────────────────────────────────────────
        if self._showing_result and self._result_amount > 0:
            amount = f"+{self._current_display_text().lstrip()}"
        else:
            amount = self._current_display_text().lstrip()
        aw = self._text_width(amount, 2, 1)
        ac = max(0, (cols - aw) // 2)

        mc = self._cascade_step if self._cascade_step >= 0 else 99999
        self._draw_dmd_text(p, amount, gx0, gy0, pitch, ac, 14, 2, 1, dot,
                            on_col, glow_col, max_col=mc)

        if self._is_winner and self._afterglow_alpha > 0.05:
            bo = QColor(on_col);    bo.setAlpha(int(self._afterglow_alpha * 255))
            bg = QColor(glow_base); bg.setAlpha(int(self._afterglow_alpha * 180))
            self._draw_dmd_text(p, amount, gx0, gy0, pitch, ac, 14, 2, 1,
                                dot * (1.0 + self._afterglow_alpha * 0.6), bo, bg)

        # ── Chase border + progress bar ───────────────────────────────────
        if self._draw_anim_active:
            chase_len = 16
            perim = (cols * 2) + (rows * 2) - 4
            head  = (self._draw_anim_frame * 3) % perim
            for step in range(chase_len):
                idx  = (head - step) % perim; fade = 1.0 - step / chase_len
                if   idx < cols:                       c2,r2 = idx, 0
                elif idx < cols+rows-1:                c2,r2 = cols-1, idx-cols+1
                elif idx < cols+rows-1+cols-1:         c2,r2 = cols-2-(idx-cols-rows+1), rows-1
                else:                                  c2,r2 = 0, rows-2-(idx-(cols+rows-1+cols-1))
                cc = QColor(on_col); cc.setAlpha(max(60, int(255*fade)))
                p.setBrush(QBrush(cc)); p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(int(gx0+c2*pitch), int(gy0+r2*pitch),
                              max(1,int(dot*1.15)), max(1,int(dot*1.15)))
            prog   = min(1.0, self._draw_anim_count / max(1, self._draw_anim_total))
            filled = int((cols - 8) * prog)
            for c2 in range(4, 4+filled):
                p.setBrush(QBrush(on_col)); p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(int(gx0+c2*pitch), int(gy0+(rows-3)*pitch),
                              max(1,int(dot)), max(1,int(dot)))
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
            border_col="#003a10", bg_col="#020a02", recess_col="#020702",
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
                prize = self._jackpot_val or jackpot_seed_for_bet(self._spots, self._bet)
            else:
                prize = table[catch] * self._bet * self._prize_mult
            rows.append((catch, prize))
        return rows

    def _draw_divider(self, p, gx0, gy0, pitch, dot, row, on_col):
        """Draw a full-width row of dim dots as a visual divider."""
        dim = QColor(on_col); dim.setAlpha(60)
        for col in range(self._dmd_cols):
            x = gx0 + col * pitch
            y = gy0 + row * pitch
            p.setBrush(QBrush(dim)); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(int(x), int(y), max(1,int(dot)), max(1,int(dot)))

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        bri  = self._breathe()

        on_base   = QColor("#00dd44");  glow_base = QColor("#008822")
        off_col   = QColor("#01100a")
        on_col    = self._scale_bri(on_base,   bri)
        glow_col  = self._scale_bri(glow_base, bri)
        hi_on     = self._scale_bri(QColor("#ffaa00"), bri)
        hi_glow   = self._scale_bri(QColor("#cc6600"),  bri)
        win_on    = QColor("#ffffff"); win_glow = QColor("#44ff88")
        dim_on    = self._scale_bri(on_base,   bri * 0.42)
        dim_glow  = self._scale_bri(glow_base, bri * 0.42)

        pitch, dot, gx0, gy0 = self._paint_base(
            p, W, H, off_col, on_col, scanline=not self._in_result)
        cols = self._dmd_cols
        rows = self._dmd_rows   # 96

        def txt(text, col, row, sc=1, gap=1, oc=None, gc=None):
            self._draw_dmd_text(p, text, gx0, gy0, pitch, col, row,
                                sc, gap, dot, oc or on_col, gc or glow_col)

        def centre(text, row, sc=1, gap=1, oc=None, gc=None):
            w = self._text_width(text, sc, gap)
            txt(text, max(0, (cols - w) // 2), row, sc, gap, oc, gc)

        def right(text, row, sc=1, gap=1, oc=None, gc=None):
            w = self._text_width(text, sc, gap)
            txt(text, max(0, cols - w - 4), row, sc, gap, oc, gc)

        def divider(row):
            self._draw_divider(p, gx0, gy0, pitch, dot, row, on_col)

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
                    co, cg = win_on, win_glow
                elif catch == self._hits and self._hits >= 0:
                    co, cg = hi_on, hi_glow
                else:
                    co, cg = on_col, glow_col
                txt(str(catch), 4, dr, oc=co, gc=cg)
                p_str = format_credits(prize)
                pw = self._text_width(p_str, 1, 1)
                txt(p_str, max(0, cols - pw - 4), dr, oc=co, gc=cg)

        # ──────────────────────────────────────────────────────────────────
        if self._in_result:
            centre("SESSION STATS", 1, oc=on_col)
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
                txt(lbl, 2, dr, 1, 1, dim_on, dim_glow)
                right(val, dr + 2, 2, 1, hi_on, hi_glow)
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
        }
        # Session stats (reset each application launch)
        self._sess_rounds: int = 0
        self._sess_total:  int = 0
        self._sess_best:   int = 0

        self._build_ui()
        self._refresh_info()

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
        lv.addWidget(self._make_board())
        lv.addWidget(self._make_drawn_row())

        mh.addWidget(left, 1)

        right_side = QWidget()
        right_side.setStyleSheet("background: transparent;")
        rv = QVBoxLayout(right_side)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(10)

        self.payout_dmd = PayoutDMD()
        self.jackpot_panel = JackpotPanel()
        rv.addWidget(self.jackpot_panel, 0)
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
        jp_val = jackpot_value_for_bet(self._jackpots, n, self.bet)
        self.payout_dmd.update_spots(n, prize_multiplier=prize_multiplier,
                                     jackpot_value=jp_val, bet=self.bet)
        self.payout_dmd.set_balance(self.balance)
        jp_tier = n if n in JACKPOT_SEEDS else None
        jp_val  = jackpot_value_for_bet(self._jackpots, n, self.bet) if jp_tier else 0
        if self.jackpot_panel._tier == jp_tier and jp_tier is not None:
            # Tier unchanged — roll the digits to the new value
            self.jackpot_panel.update_value(jp_val)
            self.jackpot_panel.setVisible(True)
        else:
            # Tier changed — snap instantly
            self.jackpot_panel.set_active_tier(jp_tier, jp_val)

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
        if not self.player_picks:
            self._set_status("⚠  SELECT AT LEAST 1 NUMBER!", "#ff6633")
            return
        if self.bet > self.balance:
            self._set_status("⚠  INSUFFICIENT CREDITS!", "#ff6633")
            return

        self.balance -= self.bet
        # Feed every pot a small % of each bet
        contrib = max(1, int(self.bet * JACKPOT_CONTRIB))
        for tier in self._jackpots:
            self._jackpots[tier] += contrib
        _save_state(self._jackpots, self.balance)
        self.jackpot_panel.stop_pulse()
        self._current_hits = 0
        self._double_up_prompted = False
        self._double_up_active = False
        self._bet_before_double_up = self.bet
        self._reset_winnings_indicator()
        self._refresh_info()
        self.play_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.random_pick_btn.setEnabled(False)

        self._reset_board_display()
        for n in self.player_picks:
            self.balls[n].set_state("selected")

        self.drawn_numbers = random.sample(range(1, 81), 20)
        self.draw_index = 0
        self._set_status("● DRAWING NUMBERS…", "#ffcc00")
        spots = len(self.player_picks)
        jp_val = jackpot_value_for_bet(self._jackpots, spots, self.bet)
        self.payout_dmd.start_draw(spots, self.bet, 1, jp_val, self.balance)
        if spots in JACKPOT_SEEDS:
            self.jackpot_panel.start_draw_animation(len(self.drawn_numbers))
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
            spots = len(self.player_picks)
            if spots in self._jackpots and self._current_hits >= spots - 2:
                self.jackpot_panel.pulse()

        # Update draw counters on every ball (hits and misses)
        self.payout_dmd.update_draw(self._current_hits, self.draw_index)

        if len(self.player_picks) in JACKPOT_SEEDS:
            self.jackpot_panel.update_draw_animation(self.draw_index, self._current_hits)

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
        answer = QMessageBox.question(
            self,
            "DOUBLE UP?",
            (
                f"Double up?\n\n"
                f"YES — pay another {format_credits(self.bet)} credits, earn bonus points\n"
                f"NO  — continue at current bet, normal payouts"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if answer == QMessageBox.StandardButton.Yes:
            self._bet_before_double_up = self.bet
            self.balance -= self.bet
            self.bet *= 2
            self._double_up_active = True
            self.credit_display.setText(format_credits(self.balance))
            self.bet_display.setText(format_credits(self.bet))
            spots = len(self.player_picks)
            jp_val = jackpot_value_for_bet(self._jackpots, spots, self.bet)
            self.payout_dmd.start_draw(spots, self.bet, 3, jp_val)
            self.payout_dmd.update_draw(self._current_hits)
            self._set_status(
                "★  DOUBLE UP ACTIVE — PAYOUT TABLE NOW SHOWS 3× WINS  ★",
                "#00ff66",
            )
        else:
            self._set_status(
                "●  DOUBLE UP DECLINED — CONTINUING DRAW",
                "#44aaff",
            )

        self.draw_timer.start(DRAW_INTERVAL_MS)

    def _finish_round(self):
        drawn_set = set(self.drawn_numbers)
        hits = len(self.player_picks & drawn_set)
        spots = len(self.player_picks)

        self.jackpot_panel.stop_pulse()
        self.jackpot_panel.stop_draw_animation()
        # Keep a final-round payout snapshot before restoring the visible bet.
        # Without this, a Double Up round briefly shows 3x during the draw, then
        # _refresh_info() rebuilds the payout table from the restored normal bet.
        final_table_bet = self.bet
        final_table_multiplier = 3 if self._double_up_active else 1
        final_table_jackpot = jackpot_value_for_bet(self._jackpots, spots, final_table_bet)

        # Check if this is a jackpot win (full catch on a progressive tier)
        jackpot_win = hits == spots and spots in self._jackpots
        jackpot_amount = 0
        if jackpot_win:
            jackpot_amount = final_table_jackpot
            self._jackpots[spots] = JACKPOT_SEEDS[spots]
            _save_state(self._jackpots, self.balance)
            self.jackpot_panel.show_winner()
            self.jackpot_panel.update_value(jackpot_seed_for_bet(spots, final_table_bet))
            final_table_jackpot = jackpot_seed_for_bet(spots, final_table_bet)

        # Payouts are bet multipliers; jackpot top-catch rows have multiplier 0
        payout_mult = PAYOUTS.get(spots, {}).get(hits, 0)
        if jackpot_win:
            payout_mult = 0   # jackpot_amount covers it entirely
        win_multiplier = 3 if self._double_up_active and (payout_mult > 0 or jackpot_win) else 1
        winnings = payout_mult * self.bet * win_multiplier + jackpot_amount
        self.balance += winnings

        if self._double_up_active:
            self.bet = self._bet_before_double_up

        # Refresh only the non-payout displays here.  The payout table is updated
        # below from the final-round snapshot so it stays on the correct 3x
        # Double Up values after both wins and losses.
        self.credit_display.setText(format_credits(self.balance))
        self.bet_display.setText(format_credits(self.bet))
        self.picks_display.setText(f"PICKS: {spots} / {self.max_picks}")
        self.picks_display.setStyleSheet(
            f"color: {'#ffcc00' if spots > 0 else '#88aacc'}; background: transparent;"
        )
        if not jackpot_win:
            jp_tier = spots if spots in JACKPOT_SEEDS else None
            self.jackpot_panel.set_active_tier(jp_tier, final_table_jackpot if jp_tier else 0)

        # Show catch result briefly on the DMD if the jackpot panel is visible
        if spots in JACKPOT_SEEDS and not jackpot_win:
            self.jackpot_panel.show_result(hits, spots, winnings)

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
                f"Jackpot resets to {format_credits(jackpot_seed_for_bet(spots, self.bet))}.",
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

        if self.balance <= 0:
            self.balance = DEFAULT_BALANCE
            self._sfx.play("broke")
            self._refresh_info()

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

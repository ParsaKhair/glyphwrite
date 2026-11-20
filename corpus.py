from dataclasses import dataclass, field

import numpy as np


# Frame per straightened line: x = col / xheight_px, y = (base_row - row) / xheight_px.
# Baseline y=0, x-height y=1, y up.


# 1 straightened line; owns all pixel data.
@dataclass(eq=False)
class Line:
    id: str
    page: str
    order: int
    segment_ids: list[str]
    # Original baseline in source pixels, to re-add drift
    baseline_poly: np.ndarray
    # Left and right edge of the line in source pixels
    x_span: tuple[float, float]
    # Px per x-height at stored resolution, to rescale for display
    xheight_px: float
    base_row: float
    # Dominant slant (dx/dy), for word gaps and glyph cuts only
    shear: float
    # Ink strength 0-1 at full source resolution, lighting normalized, never binarized
    density: np.ndarray = field(repr=False)
    # Component ID per pixel, 0 = background
    labels: np.ndarray = field(repr=False)
    # Label of the whole line, words separated by single spaces
    text: str = ""
    # "truth" (given transcription) or "ocr"
    label_source: str = "ocr"
    # What OCR read, kept when the label comes from the truth
    ocr_text: str = ""
    ocr_conf: float = np.nan
    # 1 OCR confidence per word of text
    word_confs: list[float] = field(default_factory=list)


# Run of text with no spaces; alpha = density x (labels in comp_ids).
@dataclass(eq=False)
class Segment:
    id: str
    line_id: str
    order: int
    comp_ids: list[int]
    bbox: tuple[float, float, float, float]
    # Empty if the split failed
    glyph_ids: list[str] = field(default_factory=list)
    text: str = ""
    label_source: str = "ocr"
    # NaN when the label comes from the truth
    ocr_conf: float = np.nan
    n_natural: int = 0
    count_match: bool = False
    split_ok: bool = False
    split_score: float = 0.0
    consistency: float = np.nan
    score: float = 0.0


# 1 character of a segment.
@dataclass(eq=False)
class Glyph:
    id: str
    segment_id: str
    line_id: str
    order: int
    char: str
    comp_ids: list[int]
    # Components shared with neighbours, clipped to cut
    cut_ids: list[int]
    # [left, right) in sheared x = x - y * shear
    cut: tuple[float, float]
    bbox: tuple[float, float, float, float]
    # Degrees, + leans right; measured, not corrected
    slant: float
    # Offset from baseline; measured, not corrected
    dy: float
    label_source: str = "ocr"
    ocr_conf: float = np.nan
    split_score: float = 0.0
    consistency: float = np.nan
    score: float = 0.0


# ID -> object tables.
@dataclass
class Corpus:
    lines: dict[str, Line] = field(default_factory=dict)
    segments: dict[str, Segment] = field(default_factory=dict)
    glyphs: dict[str, Glyph] = field(default_factory=dict)

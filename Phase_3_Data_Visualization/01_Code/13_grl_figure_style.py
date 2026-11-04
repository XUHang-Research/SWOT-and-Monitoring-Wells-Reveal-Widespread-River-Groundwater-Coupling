"""Shared figure typography and GRL-compatible canvas dimensions.

Use Arial for prose, numerals, and mathematical letters; unsupported symbols
may fall back to STIX. All main and supplementary figures use 9-pt text and
10-pt panel letters. The supplementary entry point is a compatibility alias
for the same typography. Mathematical scripts scale automatically.
Canvas widths remain 50--85 mm or 105--170 mm, with height at most 228 mm.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
from matplotlib.font_manager import FontProperties, findfont


MM = 1.0 / 25.4
GRL_SINGLE_COLUMN_MM = 85.0
GRL_DOUBLE_COLUMN_MM = 170.0
GRL_MAX_HEIGHT_MM = 228.0
GRL_TEXT_PT = 9.0
GRL_PANEL_PT = 10.0
SUPPLEMENT_TEXT_PT = GRL_TEXT_PT
SUPPLEMENT_PANEL_PT = GRL_PANEL_PT


def configure_grl_matplotlib() -> None:
    """Apply the common typography, symbol fallback, and export settings."""

    # Fail explicitly if the requested font is unavailable instead of silently
    # changing the appearance on another rendering machine.
    family = os.environ.get("GRL_FONT_FAMILY", "Arial")
    findfont(FontProperties(family=family), fallback_to_default=False)
    import cartopy
    maps = Path(os.environ.get("GRL_NATURAL_EARTH_DIR",
                              Path(__file__).resolve().parents[1] / "02_Resources/natural_earth"))
    if maps.is_dir():
        cartopy.config["pre_existing_data_dir"] = maps

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family],
        "font.style": "normal",
        "font.weight": "normal",
        "text.color": "#111111",
        "axes.titlecolor": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "font.size": GRL_TEXT_PT,
        "axes.labelsize": GRL_TEXT_PT,
        "axes.titlesize": GRL_TEXT_PT,
        "xtick.labelsize": GRL_TEXT_PT,
        "ytick.labelsize": GRL_TEXT_PT,
        "legend.fontsize": GRL_TEXT_PT,
        "figure.titlesize": GRL_TEXT_PT,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:style=italic",
        "mathtext.bf": f"{family}:weight=bold",
        "mathtext.bfit": f"{family}:style=italic:weight=bold",
        "mathtext.sf": family,
        "mathtext.tt": family,
        "mathtext.cal": "STIXGeneral:style=italic",
        "mathtext.fallback": "stix",
        "axes.linewidth": 0.55,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def configure_grl_supplementary_matplotlib() -> None:
    """Compatibility entry point; all figures share one 9/10-pt profile."""
    configure_grl_matplotlib()


def grl_figsize(width_mm: float, height_mm: float) -> tuple[float, float]:
    """Return inches after enforcing AGU's final-size width/height limits."""

    width = float(width_mm)
    height = float(height_mm)
    in_single_range = 50.0 <= width <= GRL_SINGLE_COLUMN_MM
    in_double_range = 105.0 <= width <= GRL_DOUBLE_COLUMN_MM
    if not (in_single_range or in_double_range):
        raise ValueError(
            f"GRL figure width must be 50--85 or 105--170 mm; got {width:g} mm"
        )
    if not 0.0 < height <= GRL_MAX_HEIGHT_MM:
        raise ValueError(
            f"GRL figure height must be >0 and <=228 mm; got {height:g} mm"
        )
    return width * MM, height * MM


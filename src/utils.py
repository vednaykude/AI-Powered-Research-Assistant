# src/utils.py
def color_for_pct(pct: float) -> str:
    # green>70 orange 40-70 red <40
    if pct >= 70:
        return "#16a34a"
    if pct >= 40:
        return "#f59e0b"
    return "#ef4444"

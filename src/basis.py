"""The single source of the basis tags shown beside every number, on screen and in the MI export."""

ASSUMED = "assumed"  # an input: an assumption, a target, the staffed capacity
SYNTHETIC = "synthetic"  # a result computed on the demo data
FROM_YOUR_DATA = "from your data"  # a result computed on an uploaded CSV

SOURCES = ("demo", "upload")


def result_basis(source: str) -> str:
    if source == "demo":
        return SYNTHETIC
    if source == "upload":
        return FROM_YOUR_DATA
    raise ValueError(f"source must be one of {SOURCES}, got {source!r}")


def label(text: str, basis: str) -> str:
    """'Staffed capacity' + 'assumed' -> 'Staffed capacity · assumed'."""
    return f"{text} · {basis}"

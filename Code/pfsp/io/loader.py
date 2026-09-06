"""Format-agnostic instance loader.

Single entry point used by the CLI, notebooks and tests to load an instance
without knowing which benchmark produced it. :func:`load_instance` determines
the benchmark format of a file and dispatches to the matching reader
(:func:`pfsp.io.taillard.read_taillard` or :func:`pfsp.io.vrf.read_vrf`),
always returning the common :class:`~pfsp.instance.Instance` type (Req 7.2).

Format detection is a cheap, robust cascade (Reqs 7.5, 7.6):

1. By file name: a ``.fsp`` extension means Taillard; a file name starting with
   ``VFR`` (the VRF naming convention, e.g. ``VFR10_5_1_Gap.txt``) means VRF.
2. Content fallback, used only when the name does not decide: the file is
   inspected so that a first numeric line of five integers (or the presence of
   Taillard text labels) is read as Taillard, while a first numeric line of two
   integers followed by body rows of even length (the ``machine time`` pairs) is
   read as VRF.
3. If neither step determines the format, ``UnknownInstanceFormatError`` is
   raised and no instance is returned (Req 7.6).
"""

from __future__ import annotations

from pathlib import PurePath

from pfsp import UnknownInstanceFormatError
from pfsp.instance import Instance
from pfsp.io.taillard import read_taillard
from pfsp.io.vrf import read_vrf

# A Taillard metadata line carries five integers (n m seed UB LB); a VRF header
# carries two (n m). These counts disambiguate the two formats by content.
_TAILLARD_HEADER_TOKEN_COUNT = 5
_VRF_HEADER_TOKEN_COUNT = 2


def _integer_tokens(line: str) -> list[int] | None:
    """Return the integers on ``line`` or ``None`` if it is not fully numeric.

    A line is numeric only when it is non-empty and every whitespace-separated
    token is an integer.
    """
    tokens = line.split()
    if not tokens:
        return None
    values: list[int] = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError:
            return None
    return values


def _detect_by_name(path: str) -> str | None:
    """Detect the benchmark from the file name alone.

    Returns ``"taillard"``, ``"vrf"`` or ``None`` when the name is not decisive.
    """
    name = PurePath(path).name
    if name.lower().endswith(".fsp"):
        return "taillard"
    if name.startswith("VFR"):
        return "vrf"
    return None


def _detect_by_content(path: str) -> str | None:
    """Detect the benchmark by inspecting the file content.

    Returns ``"taillard"``, ``"vrf"`` or ``None`` when the content is not
    decisive. Reading errors are swallowed here and surfaced later as an
    ``UnknownInstanceFormatError``; the dispatched reader reports precise
    parsing errors.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError:
        return None

    numeric_lines: list[list[int]] = []
    has_text_line = False
    for line in raw_lines:
        values = _integer_tokens(line)
        if values is None:
            if line.split():
                # A non-empty, non-numeric line is a text label.
                has_text_line = True
        else:
            numeric_lines.append(values)

    if not numeric_lines:
        return None

    first = numeric_lines[0]

    # Taillard: descriptive text labels, or a five-integer metadata header.
    if has_text_line or len(first) == _TAILLARD_HEADER_TOKEN_COUNT:
        return "taillard"

    # VRF: a two-integer "n m" header followed by body rows of even length
    # (the 2*m machine-time pairs).
    if len(first) == _VRF_HEADER_TOKEN_COUNT:
        body = numeric_lines[1:]
        if body and all(len(row) % 2 == 0 for row in body):
            return "vrf"

    return None


def load_instance(path: str) -> Instance:
    """Load a PFSP instance, detecting its benchmark format automatically.

    Parameters
    ----------
    path:
        Path to a Taillard ``.fsp`` or VRF ``VFR*`` instance file.

    Returns
    -------
    Instance
        The parsed instance, of the common :class:`~pfsp.instance.Instance`
        type regardless of the source benchmark (Req 7.2).

    Raises
    ------
    UnknownInstanceFormatError
        If the benchmark format cannot be determined from the file name or its
        content. No instance is returned in this case (Req 7.6).
    InstanceFormatError
        Propagated from the dispatched reader if the file is recognized as a
        benchmark format but cannot be parsed into a valid instance.
    """
    benchmark = _detect_by_name(path) or _detect_by_content(path)

    if benchmark == "taillard":
        return read_taillard(path)
    if benchmark == "vrf":
        return read_vrf(path)

    raise UnknownInstanceFormatError(
        f"Could not determine the benchmark format of '{path}'. Expected a "
        f"Taillard '.fsp' file or a VRF 'VFR*' file."
    )

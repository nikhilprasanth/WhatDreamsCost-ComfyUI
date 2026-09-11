"""Prompt compilation.

LTX-2.5's text encoder was trained on *flowing prose*, not tag soup. ComfyUI's
own ``TextGenerateLTX2Prompt`` system prompt is explicit about it: "Express
these as flowing prose: *a medium shot frames…, captured from a front-facing
angle as the camera slowly pans…*. Never as *medium shot, static camera —*".

So the Director's camera and lens controls do not emit keywords. They emit a
sentence. That is the whole reason this module exists: to turn structured
choices into something the encoder actually likes, while leaving the user's own
words untouched and always visible.

Nothing here constrains the model. A camera move is a phrase in the prompt, not
a patch. Where real camera-control LoRAs exist they are selected in
``spec.models.loras`` instead, and only when the installed files match the
model family.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = [
    "CAMERA_MOVES",
    "CAMERA_INTENSITY",
    "SHOT_SIZES",
    "CAMERA_ANGLES",
    "CAMERA_HEIGHTS",
    "DEPTH_OF_FIELD",
    "compile_prompt",
    "compile_camera_phrase",
    "compile_lens_phrase",
    "describe_sections",
]


# --------------------------------------------------------------------------
# vocabulary
#
# Each entry is the *predicate* of a sentence whose subject is "the camera",
# so it composes as "the camera slowly pushes in toward the subject".
# --------------------------------------------------------------------------

CAMERA_MOVES: dict[str, str] = {
    # "" is "not set" — distinct from an explicit locked-off camera. Without
    # that distinction every Director-mode prompt would end up asserting a
    # static camera the user never chose.
    "": "",
    "static": "the camera holds still on a locked-off frame",
    "pan_left": "the camera pans left across the scene",
    "pan_right": "the camera pans right across the scene",
    "tilt_up": "the camera tilts upward",
    "tilt_down": "the camera tilts downward",
    "dolly_in": "the camera dollies in toward the subject",
    "dolly_out": "the camera dollies back away from the subject",
    "truck_left": "the camera trucks left, moving laterally with the scene",
    "truck_right": "the camera trucks right, moving laterally with the scene",
    "pedestal_up": "the camera rises straight up on a pedestal",
    "pedestal_down": "the camera lowers straight down on a pedestal",
    "crane_up": "the camera cranes upward, opening out the frame",
    "crane_down": "the camera cranes downward into the scene",
    "orbit_left": "the camera orbits left around the subject",
    "orbit_right": "the camera orbits right around the subject",
    "handheld": "the handheld camera drifts and breathes with the action",
    "steadicam": "the Steadicam glides smoothly through the space",
    "tracking": "the camera tracks alongside the moving subject",
    "push_in": "the camera pushes in, tightening on the subject",
    "pull_out": "the camera pulls out, revealing more of the scene",
    "rack_focus": "focus racks from the foreground to the subject",
    "zoom_in": "the lens zooms in",
    "zoom_out": "the lens zooms out",
    "pov": "the shot is from the subject's point of view",
    "over_shoulder": "the camera sits just behind the subject's shoulder",
    "whip_pan": "the camera whip-pans to a new subject",
    "custom": "",
}

#: Adverbs that modify a move. ``custom`` defers to the user's own text.
CAMERA_INTENSITY: dict[str, str] = {
    "subtle": "almost imperceptibly",
    "moderate": "slowly",
    "strong": "quickly and decisively",
    "custom": "",
}

SHOT_SIZES: dict[str, str] = {
    "ECU": "an extreme close-up",
    "CU": "a close-up",
    "MCU": "a medium close-up",
    "MS": "a medium shot",
    "COWBOY": "a cowboy shot framed from mid-thigh",
    "FULL": "a full shot showing the whole figure",
    "WIDE": "a wide shot",
    "ESTABLISHING": "a wide establishing shot",
    "MACRO": "a macro shot",
    "POV": "a point-of-view shot",
    "OTS": "an over-the-shoulder shot",
    "TWO_SHOT": "a two-shot",
    "INSERT": "an insert shot",
}

CAMERA_ANGLES: dict[str, str] = {
    "eye": "at eye level",
    "low": "from a low angle looking up",
    "high": "from a high angle looking down",
    "dutch": "with the horizon tilted in a Dutch angle",
    "overhead": "from directly overhead",
    "worms_eye": "from a worm's-eye view at ground level",
    "front": "from a front-facing angle",
    "profile": "in profile",
    "three_quarter": "from a three-quarter angle",
    "behind": "from behind the subject",
}

CAMERA_HEIGHTS: dict[str, str] = {
    "ground": "with the camera at ground height",
    "low": "with the camera low, near knee height",
    "chest": "with the camera at chest height",
    "eye": "with the camera at eye height",
    "high": "with the camera raised above the subject",
}

DEPTH_OF_FIELD: dict[str, str] = {
    "shallow": "a shallow depth of field softening the background",
    "medium": "a moderate depth of field",
    "deep": "a deep focus holding foreground and background sharp",
    "tilt_shift": "a tilt-shift plane of focus",
}

#: Section order in the compiled prompt. Chosen so the encoder meets the
#: subject before the action, and the soundscape last — matching the shape of
#: the captions LTX was trained on.
SECTION_ORDER: tuple[str, ...] = (
    "subject", "scene", "action", "camera", "acting", "lighting", "sound",
    "dialogue", "technical",
)


# --------------------------------------------------------------------------
# phrase builders
# --------------------------------------------------------------------------

def compile_camera_phrase(camera: Any) -> str:
    """One clause describing the camera move, or "" when there is nothing to say.

    An explicit ``static`` produces a phrase — "holds still" is information the
    model uses. An *unset* move produces nothing, which is why the two are
    different values.
    """
    if camera.move == "custom" or camera.intensity == "custom":
        return camera.custom.strip()

    base = CAMERA_MOVES.get(camera.move, "")
    if not base:
        return camera.custom.strip()

    adverb = CAMERA_INTENSITY.get(camera.intensity, "")
    if not adverb or camera.move == "static":
        phrase = base
    else:
        # Insert the adverb before the verb: "the camera" + adverb + rest.
        subject, _, rest = base.partition(" ")
        if rest.startswith("camera "):
            subject, _, rest = base.partition("camera ")
            phrase = f"{subject}camera {adverb} {rest}"
        else:
            phrase = f"{subject} {adverb} {rest}"

    extra = camera.custom.strip()
    return f"{phrase}, {extra}" if extra else phrase


def compile_lens_phrase(lens: Any) -> str:
    """One sentence of framing, built only from the fields the user filled in."""
    framing = SHOT_SIZES.get(lens.shot_size, "")
    angle = CAMERA_ANGLES.get(lens.angle, "")
    height = CAMERA_HEIGHTS.get(lens.height, "")
    dof = DEPTH_OF_FIELD.get(lens.dof, "")

    parts: list[str] = []
    if framing:
        lead = f"{framing}"
        if angle:
            lead = f"{lead} {angle}"
        parts.append(lead)
    elif angle:
        parts.append(f"shot {angle}")

    if height:
        parts.append(height)
    if lens.focal_mm:
        parts.append(f"shot on a {lens.focal_mm}mm lens")
    if dof:
        parts.append(f"with {dof}")
    if lens.lighting:
        parts.append(lens.lighting.strip().rstrip("."))
    if lens.composition:
        parts.append(lens.composition.strip().rstrip("."))

    return ", ".join(parts)


def describe_sections(sections: Any) -> list[str]:
    """Non-empty Director-mode sections, in encoder-friendly order."""
    out: list[str] = []
    for name in SECTION_ORDER:
        value = getattr(sections, name, "") or ""
        value = value.strip()
        if value:
            out.append(value)
    return out


# --------------------------------------------------------------------------
# the compiler
# --------------------------------------------------------------------------

def compile_prompt(spec: Any) -> str:
    """The exact positive prompt the graph will encode.

    Three modes, three behaviours:

    ``simple``
        ``prompt.raw`` verbatim. Nothing is added, so a user who wants control
        gets it by typing.
    ``director``
        Sections, then the framing sentence, then the camera clause — joined as
        prose. The user's own section text is never rewritten, only ordered and
        punctuated.
    ``expert``
        ``prompt.raw`` verbatim, same as simple. The difference between the two
        is what the UI exposes, not what the compiler does — so switching
        between them can never change the output.

    Always inspectable: this is the function the "compiled prompt" preview
    calls, so what the user reads is what the encoder receives.
    """
    prompt = spec.prompt

    if prompt.mode in ("simple", "expert"):
        return prompt.raw.strip()

    pieces = describe_sections(prompt.sections)

    lens = compile_lens_phrase(prompt.lens)
    if lens:
        pieces.insert(0, lens)

    # The Camera section and the camera control say the same thing. When the
    # user has written the section, that is the direction — appending a
    # generated clause would restate it in slightly different words, which is
    # worse than saying it once.
    if not prompt.sections.camera.strip():
        camera = compile_camera_phrase(prompt.camera)
        if camera:
            pieces.append(camera)

    return _join_prose(pieces)


def _join_prose(pieces: Iterable[str]) -> str:
    """Join clauses into sentences.

    Capitalises the opening letter and adds terminal punctuation where the user
    left it off. That is typography, not rewriting: no word is changed, added
    or reordered within a clause.
    """
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        piece = piece[:1].upper() + piece[1:]
        if piece[-1] not in ".!?":
            piece += "."
        out.append(piece)
    return " ".join(out)

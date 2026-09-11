"""The Director Spec — the one contract everything else agrees on.

The UI edits it, the validator checks it, the compiler translates it, the
runtime nodes read it and the project file stores it. It is plain JSON: no
tensors, no base64, no waveforms. Media is a registry entry keyed by content
hash; the bytes stay on disk where ComfyUI can find them.

Time is stored in **frames** everywhere. ``TimePoint.unit`` is a display
preference only, so changing the project fps re-labels positions without
moving them.

See ``docs/ARCHITECTURE.md`` §4.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from .. import SPEC_SCHEMA, __version__
from . import ids as _ids
from .codec import from_jsonable, to_jsonable
from .time import TimeUnit, snap_dim, snap_frames

__all__ = [
    "Mode",
    "MODES",
    "RefRole",
    "REF_ROLES",
    "Anchor",
    "TimePoint",
    "Reference",
    "Segment",
    "PromptSections",
    "CameraDirection",
    "LensSetup",
    "EnhanceSettings",
    "Prompt",
    "MediaEntry",
    "AudioClip",
    "AudioSettings",
    "DecodeSettings",
    "Generation",
    "LoraRef",
    "Models",
    "RelaySettings",
    "Take",
    "ProjectSettings",
    "Meta",
    "UiState",
    "Spec",
    "new_spec",
]

# --------------------------------------------------------------------------
# enumerations, expressed as Literals so the codec validates them for free
# --------------------------------------------------------------------------

Mode = Literal[
    "t2v",        # text to video
    "i2v",        # first frame pinned
    "fflf",       # first and last frame pinned
    "keyframes",  # arbitrary pinned frames
    "continue",   # i2v seeded from a previous clip's last frame
    "a2v",        # video driven by a supplied audio track
    "t2a",        # audio only
    "iclora",     # IC-LoRA guided (control, ingredients, inpaint, …)
]
MODES: tuple[str, ...] = (
    "t2v", "i2v", "fflf", "keyframes", "continue", "a2v", "t2a", "iclora",
)

RefRole = Literal[
    "keyframe",     # a pinned frame on the timeline
    "character",    # identity reference (reference-sheet / ingredients)
    "environment",
    "object",
    "style",
    "motion",       # driving video for IC-LoRA motion / control
    "control",      # depth / canny / pose annotation
]
REF_ROLES: tuple[str, ...] = (
    "keyframe", "character", "environment", "object", "style", "motion", "control",
)

#: ``start`` and ``end`` pin to frame 0 and frame -1 respectively; ``index``
#: uses :attr:`Reference.at`. Keeping them distinct means a last-frame anchor
#: survives a duration change instead of drifting off the end.
Anchor = Literal["start", "end", "index"]

PromptMode = Literal["simple", "director", "expert"]
SeedMode = Literal["fixed", "random", "increment", "decrement", "list"]
AudioMode = Literal["generate", "import", "inpaint", "mute"]
FitMode = Literal["cover", "contain", "stretch"]
CropMode = Literal["center", "disabled"]
LoraKind = Literal["lora", "iclora"]


# --------------------------------------------------------------------------
# leaves
# --------------------------------------------------------------------------

@dataclass
class TimePoint:
    """A position on the timeline.

    ``frame`` is the truth. ``unit`` records how the user prefers to read it,
    so the inspector can show ``2.42 s`` or ``58`` or ``48%`` without any of
    them becoming the stored value.
    """

    frame: int = 0
    unit: TimeUnit = "seconds"


@dataclass
class Reference:
    """One image or clip attached to the shot."""

    id: str = ""
    role: RefRole = "keyframe"
    media: str | None = None
    label: str = ""
    enabled: bool = True
    at: TimePoint = field(default_factory=TimePoint)
    anchor: Anchor = "index"
    strength: float = 1.0
    fit: FitMode = "cover"
    crop: CropMode = "center"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _ids.new_id("reference")

    def resolved_frame(self, total_frames: int) -> int:
        """The frame index this reference compiles to.

        ``end`` resolves to ``-1`` because that is what ``LTXVAddGuide``
        documents for a last-frame anchor; the node counts negatives from the
        end, which keeps the anchor correct if the duration changes later.
        """
        if self.anchor == "start":
            return 0
        if self.anchor == "end":
            return -1
        return max(0, min(self.at.frame, max(0, total_frames - 1)))


@dataclass
class Segment:
    """A Prompt Relay region: a stretch of the timeline with its own prompt."""

    id: str = ""
    start: int = 0
    length: int = 1
    text: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _ids.new_id("segment")

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class PromptSections:
    """Director-mode structure. Every field is optional; empty ones are skipped."""

    subject: str = ""
    scene: str = ""
    action: str = ""
    camera: str = ""
    acting: str = ""
    lighting: str = ""
    sound: str = ""
    dialogue: str = ""
    technical: str = ""


@dataclass
class CameraDirection:
    """Camera movement, compiled into prose rather than forced on the model.

    ``move`` defaults to "" meaning *unset*, which is deliberately not the same
    as ``"static"``: an untouched camera control must add nothing to the prompt,
    while an explicitly locked-off camera is a direction worth stating.
    """

    move: str = ""
    intensity: str = "moderate"
    custom: str = ""


@dataclass
class LensSetup:
    """Optional cinematography metadata that shapes the compiled prompt."""

    shot_size: str = ""
    angle: str = ""
    height: str = ""
    focal_mm: int = 0
    dof: str = ""
    lighting: str = ""
    composition: str = ""


@dataclass
class EnhanceSettings:
    """``TextGenerateLTX2Prompt`` settings. Off by default — it costs a model load."""

    enabled: bool = False
    seed: int = 0
    max_length: int = 600
    use_image: bool = True


@dataclass
class Prompt:
    mode: PromptMode = "simple"
    raw: str = ""
    negative: str = ""
    sections: PromptSections = field(default_factory=PromptSections)
    camera: CameraDirection = field(default_factory=CameraDirection)
    lens: LensSetup = field(default_factory=LensSetup)
    enhance: EnhanceSettings = field(default_factory=EnhanceSettings)


@dataclass
class MediaEntry:
    """A file the project points at. Never the file's contents."""

    id: str = ""
    filename: str = ""
    subfolder: str = ""
    kind: Literal["image", "video", "audio"] = "image"
    sha256: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0
    size: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _ids.new_id("media")

    @property
    def path(self) -> str:
        """Path relative to ComfyUI's input directory."""
        return f"{self.subfolder}/{self.filename}" if self.subfolder else self.filename


@dataclass
class AudioClip:
    id: str = ""
    media: str | None = None
    start: int = 0
    length: int = 1
    trim_start: float = 0.0
    gain: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _ids.new_id("audio")

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class AudioSettings:
    enabled: bool = True
    mode: AudioMode = "generate"
    clips: list[AudioClip] = field(default_factory=list)
    #: Media id for ``LTXVReferenceAudio`` speaker-identity transfer.
    reference: str | None = None
    identity_guidance: float = 3.0


@dataclass
class DecodeSettings:
    """``VAEDecodeTiled`` parameters. Larger tiles are faster and need more VRAM."""

    tile_size: int = 512
    overlap: int = 64
    temporal_size: int = 64
    temporal_overlap: int = 8
    bit_depth: int = 8


@dataclass
class Generation:
    preset: str = "balanced"
    stages: int = 1
    seed: int = 42
    seed_mode: SeedMode = "fixed"
    seed_list: list[int] = field(default_factory=list)
    video_cfg: float = 1.0
    audio_cfg: float = 1.0
    sampler: str = "euler_ancestral"
    refine_sampler: str = "euler"
    #: Explicit sigma schedules. LTX distilled models ship fixed schedules
    #: rather than a step count, so these are the schedule, not a hint.
    sigmas_stage1: list[float] = field(default_factory=list)
    sigmas_stage2: list[float] = field(default_factory=list)
    img_compression: int = 18
    first_frame_strength: float = 0.7
    guide_strength: float = 0.7
    decode: DecodeSettings = field(default_factory=DecodeSettings)
    #: Stage-1 spatial divisor for two-stage runs (the official graphs halve it).
    stage1_divisor: int = 2
    save_prefix: str = "video/LTXDirector"


@dataclass
class LoraRef:
    name: str = ""
    strength: float = 1.0
    kind: LoraKind = "lora"
    enabled: bool = True


@dataclass
class Models:
    family: str = "ltx2.5"
    unet: str = ""
    vae: str = ""
    audio_vae: str = ""
    clip: str = ""
    enhancer_clip: str | None = None
    upscaler: str | None = None
    loras: list[LoraRef] = field(default_factory=list)


@dataclass
class RelaySettings:
    """Prompt Relay knobs. ``epsilon`` controls how sharp segment boundaries are."""

    enabled: bool = True
    epsilon: float = 0.001
    video_strength: float = 1.0
    audio_strength: float = 1.0
    video_window_scale: float = 1.0
    audio_window_scale: float = 1.0


@dataclass
class Take:
    """One generation, recorded so it can be compared, re-run or promoted."""

    id: str = ""
    seed: int = 0
    starred: bool = False
    created: str = ""
    output: str = ""
    settings_digest: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _ids.new_id("take")


@dataclass
class ProjectSettings:
    fps: float = 24.0
    width: int = 1280
    height: int = 704
    frames: int = 121
    mode: Mode = "t2v"

    def normalise(self) -> None:
        """Snap geometry to what LTX accepts. Safe to call repeatedly."""
        self.width = snap_dim(self.width)
        self.height = snap_dim(self.height)
        self.frames = snap_frames(self.frames)
        if self.fps <= 0:
            self.fps = 24.0

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


@dataclass
class Meta:
    name: str = "Untitled shot"
    created: str = ""
    modified: str = ""
    director_version: str = __version__
    note: str = ""


@dataclass
class UiState:
    """Purely cosmetic. Excluded from the settings digest so it never dirties a take."""

    display_unit: TimeUnit = "seconds"
    expanded: list[str] = field(default_factory=list)
    zoom: float = 1.0
    playhead: int = 0
    selection: str = ""


# --------------------------------------------------------------------------
# root
# --------------------------------------------------------------------------

#: Top-level keys that describe presentation or history rather than the shot,
#: and so must not change the digest used to key caches and takes.
DIGEST_EXCLUDE: tuple[str, ...] = ("ui", "takes", "meta")


@dataclass
class Spec:
    schema: int = SPEC_SCHEMA
    meta: Meta = field(default_factory=Meta)
    project: ProjectSettings = field(default_factory=ProjectSettings)
    prompt: Prompt = field(default_factory=Prompt)
    references: list[Reference] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    audio: AudioSettings = field(default_factory=AudioSettings)
    media: dict[str, MediaEntry] = field(default_factory=dict)
    generation: Generation = field(default_factory=Generation)
    models: Models = field(default_factory=Models)
    relay: RelaySettings = field(default_factory=RelaySettings)
    takes: list[Take] = field(default_factory=list)
    ui: UiState = field(default_factory=UiState)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_json(self, *, indent: int | None = None) -> str:
        """Stable JSON. Keys follow declaration order, so diffs stay readable."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            separators=(",", ":") if indent is None else (",", ": "),
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Spec":
        """Build a Spec, migrating older payloads on the way in."""
        if not data:
            return cls()
        from .migrate import migrate  # local import: migrate imports Spec

        return from_jsonable(cls, migrate(data))

    @classmethod
    def from_json(cls, text: str | None) -> "Spec":
        """Parse a Spec from a widget value or file.

        Empty or unparseable text yields a default Spec rather than raising:
        an empty node is a normal state, not an error.
        """
        if not text or not text.strip():
            return cls()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)

    def copy(self) -> "Spec":
        return Spec.from_dict(self.to_dict())

    # -- derived -----------------------------------------------------------

    def digest(self) -> str:
        """Digest of everything that affects the output."""
        return _ids.digest_obj(self.to_dict(), exclude=DIGEST_EXCLUDE)

    def normalise(self) -> "Spec":
        """Apply the invariants the rest of the system assumes. Returns self."""
        self.project.normalise()
        self.schema = SPEC_SCHEMA
        for ref in self.references:
            ref.at.frame = max(0, min(ref.at.frame, max(0, self.project.frames - 1)))
            ref.strength = _clamp(ref.strength, 0.0, 1.0)
        for clip in self.audio.clips:
            clip.start = max(0, clip.start)
            clip.length = max(1, clip.length)
        self.segments.sort(key=lambda s: s.start)
        return self

    # -- lookups -----------------------------------------------------------

    def media_for(self, media_id: str | None) -> MediaEntry | None:
        return self.media.get(media_id) if media_id else None

    def active_references(self, *roles: str) -> list[Reference]:
        """Enabled references with resolvable media, optionally filtered by role.

        Sorted by anchor then frame so ``start`` anchors compile before indexed
        ones and ``end`` anchors last — the order ``LTXVAddGuide`` chains expect.
        """
        order = {"start": 0, "index": 1, "end": 2}
        out = [
            r for r in self.references
            if r.enabled and r.media and r.media in self.media
            and (not roles or r.role in roles)
        ]
        out.sort(key=lambda r: (order[r.anchor], r.at.frame))
        return out

    @property
    def relay_active(self) -> bool:
        """True when there is genuinely more than one prompt region to relay."""
        return self.relay.enabled and len([s for s in self.segments if s.text.strip()]) > 1

    @property
    def needs_audio(self) -> bool:
        """Audio-only mode always needs audio; otherwise it is the user's choice."""
        if self.project.mode == "t2a":
            return True
        return self.audio.enabled and self.audio.mode != "mute"

    @property
    def needs_video(self) -> bool:
        return self.project.mode != "t2a"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


#: The distilled 8-step schedule every LTX-2.5 example graph uses.
DEFAULT_SIGMAS_STAGE1: tuple[float, ...] = (
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
)

#: The 3-step refine schedule used after ``LTXVLatentUpsampler``.
DEFAULT_SIGMAS_STAGE2: tuple[float, ...] = (0.85, 0.725, 0.4219, 0.0)


def new_spec(**overrides: Any) -> Spec:
    """A fresh Spec with LTX-2.5 defaults, for the UI's "new project" action."""
    spec = Spec()
    spec.generation.sigmas_stage1 = list(DEFAULT_SIGMAS_STAGE1)
    spec.generation.sigmas_stage2 = list(DEFAULT_SIGMAS_STAGE2)
    for key, value in overrides.items():
        if not hasattr(spec, key):
            raise ValueError(f"Spec has no field {key!r}.")
        setattr(spec, key, value)
    return spec.normalise()


def fields_of(cls: type) -> tuple[str, ...]:
    """Field names of a spec dataclass — used by the frontend schema mirror."""
    return tuple(f.name for f in dataclasses.fields(cls))

from __future__ import annotations

from copy import deepcopy
from typing import Any

SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "nl", "name": "Dutch"},
    {"code": "srn", "name": "Sranan Tongo"},
    {"code": "hns", "name": "Sarnami Hindustani"},
    {"code": "hi-Latn", "name": "Hindi (Romanized)"},
    {"code": "jv", "name": "Javanese"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "pap", "name": "Papiamento"},
]

CULTURE_PROFILES: dict[str, dict[str, Any]] = {
    "suriname-kaseko": {
        "name": "Surinamese Kaseko",
        "region": "Suriname",
        "tempo_bpm": [108, 138],
        "meter": "4/4",
        "rhythm": "syncopated snare and bass-drum dialogue, lively brass punctuations, call-and-response phrasing",
        "instruments": ["skratji drum", "bass drum", "saxophone", "trumpet", "trombone", "electric bass", "guitar", "vocals"],
        "languages": ["srn", "nl", "hns"],
        "production_notes": "Keep percussion forward, preserve human push-and-pull, and avoid quantizing every hit to a rigid grid.",
        "provenance": "Curated style metadata only; no claim is made that a trained Kaseko model is bundled.",
    },
    "suriname-kawina": {
        "name": "Surinamese Kawina",
        "region": "Suriname",
        "tempo_bpm": [92, 126],
        "meter": "4/4",
        "rhythm": "interlocking hand-drum patterns, strong call-and-response, cyclical grooves and communal vocal energy",
        "instruments": ["hand drums", "shakers", "claves", "chorus vocals", "lead vocals", "bass"],
        "languages": ["srn", "nl"],
        "production_notes": "Use layered live percussion and responsive chorus phrasing rather than a generic dancehall beat.",
        "provenance": "Curated style metadata only; no claim is made that a trained Kawina model is bundled.",
    },
    "suriname-baithak-gana": {
        "name": "Surinamese Baithak Gana",
        "region": "Suriname",
        "tempo_bpm": [78, 122],
        "meter": "4/4 or 6/8",
        "rhythm": "dholak-driven Hindustani-Caribbean groove with harmonium support and conversational melodic ornamentation",
        "instruments": ["dholak", "harmonium", "dhantal", "vocals", "handclaps", "bass"],
        "languages": ["hns", "hi-Latn", "nl"],
        "production_notes": "Retain dholak articulation, dhantal pulse and lyrical phrasing associated with Sarnami performance practice.",
        "provenance": "Curated style metadata only; no claim is made that a trained Baithak Gana model is bundled.",
    },
    "caribbean-soca": {
        "name": "Soca",
        "region": "Caribbean",
        "tempo_bpm": [125, 165],
        "meter": "4/4",
        "rhythm": "driving kick, syncopated snare, rolling percussion, bright rhythmic synths and carnival lift",
        "instruments": ["drum kit", "congas", "iron", "brass", "synth bass", "rhythmic synths", "vocals"],
        "languages": ["en", "fr", "es"],
        "production_notes": "Build energy through percussion density, rhythmic hooks and crowd-response sections.",
        "provenance": "Curated style metadata only; no proprietary artist model is included.",
    },
    "caribbean-calypso": {
        "name": "Calypso",
        "region": "Caribbean",
        "tempo_bpm": [88, 126],
        "meter": "4/4",
        "rhythm": "light syncopation, melodic bass movement, conversational verses and responsive brass or guitar figures",
        "instruments": ["acoustic guitar", "electric bass", "brass", "hand percussion", "steelpan", "vocals"],
        "languages": ["en", "fr"],
        "production_notes": "Prioritize storytelling, lyrical clarity and rhythmic wit over dense modern mastering.",
        "provenance": "Curated style metadata only; no proprietary artist model is included.",
    },
    "caribbean-reggae": {
        "name": "Reggae",
        "region": "Caribbean",
        "tempo_bpm": [68, 96],
        "meter": "4/4",
        "rhythm": "one-drop or steppers drum feel, offbeat guitar skank, melodic bass and spacious dub-aware arrangement",
        "instruments": ["drums", "electric bass", "guitar", "organ", "piano", "horns", "vocals"],
        "languages": ["en", "nl", "srn"],
        "production_notes": "Leave space between elements and preserve bass depth without crushing transients.",
        "provenance": "Curated style metadata only; no proprietary artist model is included.",
    },
    "caribbean-dancehall": {
        "name": "Dancehall",
        "region": "Caribbean",
        "tempo_bpm": [88, 112],
        "meter": "4/4",
        "rhythm": "syncopated digital riddim, sparse but forceful low end, clipped percussion and vocal-forward arrangement",
        "instruments": ["drum machine", "sub bass", "synth stabs", "percussion", "vocals"],
        "languages": ["en", "nl", "srn"],
        "production_notes": "Keep the riddim lean and leave enough spectral space for lead vocals and ad-libs.",
        "provenance": "Curated style metadata only; no proprietary artist model is included.",
    },
    "caribbean-zouk-kompa": {
        "name": "Zouk / Kompa",
        "region": "Caribbean",
        "tempo_bpm": [88, 124],
        "meter": "4/4",
        "rhythm": "smooth syncopated guitar, steady dance pulse, melodic bass, warm keyboards and romantic vocal phrasing",
        "instruments": ["electric guitar", "bass", "drum kit", "keyboards", "horns", "vocals"],
        "languages": ["fr", "en", "nl"],
        "production_notes": "Use fluid guitar comping and controlled percussion rather than generic four-on-the-floor EDM.",
        "provenance": "Curated style metadata only; no proprietary artist model is included.",
    },
    "caribbean-chutney": {
        "name": "Chutney / Chutney Soca",
        "region": "Caribbean",
        "tempo_bpm": [108, 154],
        "meter": "4/4",
        "rhythm": "dholak and tassa-derived rhythmic drive combined with Caribbean bass, synth and carnival arrangement",
        "instruments": ["dholak", "tassa", "dhantal", "harmonium", "synth", "bass", "vocals"],
        "languages": ["hns", "hi-Latn", "en"],
        "production_notes": "Keep Indo-Caribbean percussion and melodic ornamentation audible instead of reducing the style to generic soca.",
        "provenance": "Curated style metadata only; no proprietary artist model is included.",
    },
}


def list_profiles() -> list[dict[str, Any]]:
    return [{"id": profile_id, **deepcopy(profile)} for profile_id, profile in CULTURE_PROFILES.items()]


def get_profile(profile_id: str | None) -> dict[str, Any] | None:
    if not profile_id:
        return None
    profile = CULTURE_PROFILES.get(profile_id)
    return deepcopy(profile) if profile else None


def enhance_prompt(prompt: str, profile_id: str | None, language: str | None = None) -> str:
    profile = get_profile(profile_id)
    if not profile:
        return prompt.strip()
    language_text = f"Primary lyric language: {language}. " if language else ""
    instruments = ", ".join(profile["instruments"])
    low, high = profile["tempo_bpm"]
    return (
        f"{prompt.strip()}\n\n"
        f"Cultural production profile: {profile['name']} from {profile['region']}. "
        f"Target tempo range {low}-{high} BPM, meter {profile['meter']}. "
        f"Rhythmic identity: {profile['rhythm']}. "
        f"Core instrumentation: {instruments}. "
        f"{language_text}Production guidance: {profile['production_notes']} "
        "Respect the tradition, avoid parody, and do not imitate any named living artist."
    )

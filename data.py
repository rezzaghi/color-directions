# Dataset generator. Run `python data.py`: runs sanity checks, writes data/*.json.
# Edit the templates here, never the JSON by hand.
import json
import re
from pathlib import Path

COLORS = ["red", "blue", "green"]

# Explicit set: minimal pairs, only the color word changes. Colorless objects, no idioms,
# color never sentence-initial. We read the color token.
EXPLICIT_TEMPLATES = [
    "She wore a {c} dress to the party.",
    "He painted the door {c}.",
    "They bought a {c} car last year.",
    "I drank my coffee from a {c} mug.",
    "The child was holding a {c} balloon.",
    "Please sign the form with the {c} pen.",
    "He always wears a {c} shirt on Fridays.",
    "My sister rides a {c} bike to school.",
    "We sat on the {c} sofa in the living room.",
    "Put the receipts in the {c} folder.",
    "She wrote everything down in a {c} notebook.",
    "I forgot my {c} umbrella on the bus.",
    "He wrapped a {c} scarf around his neck.",
    "The kids filled the {c} bucket with sand.",
    "She dried her hands on a {c} towel.",
    "The teacher corrected the test with a {c} marker.",
    "The gift came in a {c} box.",
    "She tied a {c} ribbon around the package.",
    "He carried a {c} backpack on the trip.",
    "A {c} lamp stood in the corner of the room.",
    "My grandmother knitted me a {c} sweater.",
    "He lost his {c} hat at the concert.",
    "The children flew a {c} kite at the beach.",
    "The letter arrived in a {c} envelope.",
    "I bought a new {c} toothbrush.",
    "They put a {c} rug in the hallway.",
    "She hung {c} curtains in the bedroom.",
    "The bathroom walls were covered in {c} tiles.",
    "He wore {c} gloves while gardening.",
    "I can't find my other {c} sock.",
    "They painted the fence {c} last summer.",
    "A small {c} boat was tied to the dock.",
    "The child picked the {c} crayon from the box.",
    "One {c} button was missing from his coat.",
    "She dyed her hair {c} for the festival.",
    "The walls of the office were painted {c}.",
    "He keeps his tools in a {c} toolbox.",
    "She bought a {c} phone case online.",
    "The restaurant had {c} chairs on the terrace.",
    "He drove a {c} truck to work.",
]

EXPLICIT = [
    {"text": tpl.format(c=c), "target": c, "label": c, "group": i}
    for i, tpl in enumerate(EXPLICIT_TEMPLATES)
    for c in COLORS
]

# Implicit set (test only): no color words. Each category has one red, one blue, one green
# object, so category carries no color information. We read the object's last token.
IMPLICIT_TRIPLES = {
    # category:     (red,                  blue,                green)
    "prototype1":   ("blood",              "the clear sky",     "fresh grass"),
    "prototype2":   ("a ripe tomato",      "the open ocean",    "a fresh lime"),
    "gemstone1":    ("a ruby",             "a sapphire",        "an emerald"),
    "gemstone2":    ("a garnet",           "lapis lazuli",      "jade"),
    "muppet":       ("Elmo",               "Cookie Monster",    "Kermit the Frog"),
    "videogame":    ("Mario",              "Sonic the Hedgehog", "Yoshi"),
    "logo":         ("the Coca-Cola logo", "the Facebook logo", "the Starbucks logo"),
    "manmade":      ("a fire truck",       "a pair of jeans",   "a dollar bill"),
    "character":    ("Santa Claus",        "a Smurf",           "the Grinch"),
    "plant":        ("a rose",             "a cornflower",      "a fern"),
}

IMPLICIT_FRAMES = [
    "I kept thinking about {o}.",
    "There was a picture of {o} on the wall.",
    "He told me a story about {o}.",
    "She pointed at {o} and smiled.",
]

IMPLICIT = [
    {"text": fr.format(o=obj), "target": obj, "label": color, "group": cat}
    for cat, objs in IMPLICIT_TRIPLES.items()
    for color, obj in zip(COLORS, objs)
    for fr in IMPLICIT_FRAMES
]

# Controls: objects with no typical color.
CONTROL_OBJECTS = [
    "a chair", "a spoon", "a book", "a window", "a table", "a key",
    "a clock", "a bicycle", "a laptop", "a pillow", "a suitcase", "a bottle",
]

CONTROLS = [
    {"text": fr.format(o=obj), "target": obj, "label": "none", "group": "control"}
    for obj in CONTROL_OBJECTS
    for fr in IMPLICIT_FRAMES
]


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print(f"explicit: {len(EXPLICIT)}  implicit: {len(IMPLICIT)}  controls: {len(CONTROLS)}")

    for ex in EXPLICIT + IMPLICIT + CONTROLS:
        assert ex["target"] in ex["text"], ex

    # No color word may leak into implicit/control sentences ("blueberry" counts).
    leak = re.compile(r"\bred\b|blue|green|scarlet|crimson|azure|navy", re.I)
    for ex in IMPLICIT + CONTROLS:
        assert not leak.search(ex["text"]), ex["text"]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    for c in COLORS:
        assert len(tok.encode(" " + c, add_special_tokens=False)) == 1
    for cat, objs in IMPLICIT_TRIPLES.items():
        print(f"{cat:11s}", " | ".join(str(tok.tokenize(" " + o)) for o in objs))
    print("all checks passed")

    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    for name, examples in [("explicit", EXPLICIT), ("implicit", IMPLICIT), ("controls", CONTROLS)]:
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=4, ensure_ascii=False)
        print(f"wrote data/{name}.json ({len(examples)} examples)")

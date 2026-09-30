# Color directions in Qwen2.5-1.5B

**Does a language model know that a tomato is red, even when the word "red" never appears?**

This is a small learning experiment in mechanistic interpretability. It finds a "red", "blue" and "green" direction inside [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) using sentences that *contain* color words, then checks whether sentences about colored objects ("a ruby", "the clear sky", "a fern") point along those same directions.

**Result:** at layer 12, **24 of 30 objects** are assigned the right color by directions that were computed only from sentences like "She wore a red dress." Chance would be 10 of 30.

## The idea in one picture

Inside the model, every token gets a vector (here, 1536 numbers), called the **residual stream**. That vector holds everything the model currently "thinks" about the token: which word it is, its grammar, the rest of the sentence… and perhaps its color.

The plan:

1. Run sentences that differ **only** in the color word ("a red dress" / "a blue dress" / "a green dress") and record the vector at the color word.
2. Average and subtract to get one **direction** per color.
3. Record the vector of sentences **without any color word** ("I kept thinking about a ruby.") and check which color direction it points along.

## How it works, step by step

All numbers below are real: layer 12 of Qwen2.5-1.5B-Instruct, using the full dataset. A real vector has 1536 numbers, so only the first 4 are shown, and `…` stands for the other 1532. All the math uses all 1536.

### The data

120 sentences: 40 templates × 3 colors, where only the color word changes. Two of the templates, A = "She wore a ___ dress to the party." and B = "He painted the door ___.", give these vectors at the color word:

```
A-red   = [+0.87, -0.28, -2.12, -0.64, …]     "She wore a red dress to the party."
B-red   = [+1.16, -0.54, -1.98, -0.22, …]     "He painted the door red."   ← different from A-red
A-blue  = [+2.66, -0.32, -1.48, -0.86, …]
B-blue  = [+2.62, -0.28, -1.20, -0.41, …]
A-green = [-3.28, -0.27, -1.32, -0.53, …]
B-green = [-2.31, -0.30, -1.12, -0.12, …]
… + 38 more templates per color
```

- Each vector mixes the **sentence part** (dress or door, grammar, position…) with the **color part**. In a real model the two aren't in separate numbers; they're blended into all 1536.
- The red vectors differ from each other, because their sentences differ.

### Step 1: mean per color (number by number, over 40 sentences)

```
mean(red)   = (A-red + B-red + … all 40 red vectors) / 40       = [+1.24, -0.10, -1.31, -0.27, …]
mean(blue)  = (A-blue + B-blue + … all 40 blue vectors) / 40    = [+2.26, -0.29, -0.77, -0.40, …]
mean(green) = (A-green + B-green + … all 40 green vectors) / 40 = [-3.90, -0.17, -0.79, -0.34, …]
```

- Every template appears once per color, so the sentence part averages to the same "average sentence" in all three means.
- The differences between individual red sentences average out.

### Step 2: grand mean

```
grand = (mean(red) + mean(blue) + mean(green)) / 3
      = [ (1.24+2.26−3.90)/3, (−0.10−0.29−0.17)/3, (−1.31−0.77−0.79)/3, (−0.27−0.40−0.34)/3, … ]
      = [ -0.13, -0.19, -0.96, -0.34, … ]
```

- This is "the average sentence, with the colors averaged away".

### Step 3: subtract → directions

```
red dir   = [+1.24, -0.10, -1.31, -0.27, …] − [-0.13, -0.19, -0.96, -0.34, …] = [+1.37, +0.09, -0.35, +0.07, …]
blue dir  = [+2.26, -0.29, -0.77, -0.40, …] − [-0.13, -0.19, -0.96, -0.34, …] = [+2.39, -0.11, +0.18, -0.07, …]
green dir = [-3.90, -0.17, -0.79, -0.34, …] − [-0.13, -0.19, -0.96, -0.34, …] = [-3.76, +0.02, +0.17, -0.00, …]
```

- The shared sentence part is removed: numbers 2–4 drop close to 0, while number 1, which carries a lot of color, stays large.
- What's left in each direction is only how that color differs from the average color.

### Step 4: normalize to length 1

Only the direction matters, not the length, so we divide each by its length (the square root of the sum of all 1536 squared numbers):

```
|red dir|   = √(1.37² + 0.09² + 0.35² + 0.07² + … 1532 more) = √335.6 = 18.32 → red_hat   = [+0.07, +0.00, -0.02, +0.00, …]
|blue dir|  = √(2.39² + 0.11² + 0.18² + 0.07² + … 1532 more) = √316.8 = 17.80 → blue_hat  = [+0.13, -0.01, +0.01, -0.00, …]
|green dir| = √(3.76² + 0.02² + 0.17² + 0.00² + … 1532 more) = √328.0 = 18.11 → green_hat = [-0.21, +0.00, +0.01, -0.00, …]

e.g. red_hat = [ 1.37/18.32, 0.09/18.32, −0.35/18.32, 0.07/18.32, … ]
```

- Without this step, a color with a longer arrow would win more often just because of its length. Here the three lengths are similar (about 18), so it's a safeguard.
- The numbers are small because a total length of 1 is shared across 1536 numbers.

### Step 5: classify by projecting

For each vector: subtract grand, take the dot product with each unit direction, and pick the largest. Each score is a sum of 1536 products (the first 4 are shown).

```
A-red → A-red − grand = [+1.00, -0.09, -1.17, -0.30, …]
red:   (1.00)(0.07) + (−0.09)(0.00) + (−1.17)(−0.02) + (−0.30)(0.00) + … = +20.74  ← highest ✓
blue:  (1.00)(0.13) + (−0.09)(−0.01) + (−1.17)(0.01) + (−0.30)(−0.00) + … = −10.40
green: (1.00)(−0.21) + (−0.09)(0.00) + (−1.17)(0.01) + (−0.30)(−0.00) + … = −10.76

B-red → B-red − grand = [+1.30, -0.36, -1.03, +0.12, …]
red: +17.73 ✓    blue: −8.56    green: −9.52

A-blue → A-blue − grand = [+2.79, -0.13, -0.52, -0.52, …]
red: −8.77    blue: +17.73 ✓    green: −8.55
```

Over all 120 sentences, the red score ranges from +16.59 to +20.74 for red sentences, and from −11.49 to −6.60 for blue and green ones. The scores vary from sentence to sentence, but there's no overlap.

### The implicit test: ocean and fire

The directions are now frozen. These sentences contain no color word; the model has to supply the color itself:

```
ocean = "I kept thinking about the ocean."  → ocean − grand = [+4.54, -0.65, +0.73, +0.26, …]
red:   (4.54)(0.07) + (−0.65)(0.00) + (0.73)(−0.02) + (0.26)(0.00) + … = −6.46
blue:  (4.54)(0.13) + (−0.65)(−0.01) + (0.73)(0.01) + (0.26)(−0.00) + … = +3.80  ✓
green: (4.54)(−0.21) + (−0.65)(0.00) + (0.73)(0.01) + (0.26)(−0.00) + … = +2.80

fire = "I kept thinking about a fire."      → fire − grand  = [+2.85, -0.53, +0.21, +0.60, …]
red:   (2.85)(0.07) + (−0.53)(0.00) + (0.21)(−0.02) + (0.60)(0.00) + … = +8.38  ✓
blue:  (2.85)(0.13) + (−0.53)(−0.01) + (0.21)(0.01) + (0.60)(−0.00) + … = −7.21
green: (2.85)(−0.21) + (−0.53)(0.00) + (0.21)(0.01) + (0.60)(−0.00) + … = −1.39
```

Ocean is classified as **blue** (with green close behind; the ocean is often blue-green), and fire as **red**. Neither sentence contained a color word, and their sentence parts are new, but the directions still read the right color.

## Full results

The full test uses **30 objects** in 10 categories. Each category has one red, one blue and one green object, so the category itself says nothing about color. Every object appears in 4 neutral sentences ("I kept thinking about ___.", "There was a picture of ___ on the wall.", …). For this run the directions are computed from 30 of the 40 templates; the other 10 are held out as a check.

**Accuracy by layer** (chance = 0.33):

| layer | 0 | 3 | 6 | 9 | **12** | 15 | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| implicit objects | 0.63 | 0.70 | 0.76 | 0.76 | **0.80** | 0.75 | 0.75 | 0.77 | 0.74 | 0.71 |

**Object by object at layer 12** (scores averaged over the 4 sentences):

| category | red | blue | green |
|---|---|---|---|
| prototype | blood ✅ | the clear sky ✅ | fresh grass ✅ |
| prototype | a ripe tomato ✅ | the open ocean ✅ | a fresh lime ✅ |
| gemstone | a ruby ✅ | a sapphire ✅ | an emerald ✅ |
| gemstone | a garnet ✅ | lapis lazuli ✅ | jade ✅ |
| plant | a rose ✅ | a cornflower ✅ | a fern ✅ |
| man-made | a fire truck ✅ | a pair of jeans ✅ | a dollar bill ❌ (blue) |
| character | Santa Claus ✅ | a Smurf ✅ | the Grinch ✅ |
| logo | the Coca-Cola logo ✅ | the Facebook logo ✅ | the Starbucks logo ❌ (red) |
| muppet | Elmo ❌ (blue) | Cookie Monster ✅ | Kermit the Frog ✅ |
| video game | Mario ❌ (green) | Sonic the Hedgehog ❌ (red) | Yoshi ❌ (red) |

**24 / 30 correct.** Natural things and materials (prototypes, gemstones, plants) are all correct with strong scores. Fictional characters and brands are weaker, and the video game characters all fail. Either the model knows less about them, or they really are multi-colored (Mario wears red *and* blue).

## Honest caveats

- **Much of the signal is already in the word itself.** Layer 0 is just the word's embedding, before any context, and it already scores 0.63. The embedding of "sky" sits closer to "blue" simply because the words appear together in text. Context adds the rest (up to 0.80).
- **Centering matters.** Color words are adjectives and objects are nouns, and the model represents the two differently. That shared "noun" offset can push every object toward one color. The implicit vectors are therefore centered on their own mean. This is allowed because that set has equally many red, blue and green objects, so its mean carries no color.
- **Layer 12 was picked by looking at the results**, which makes it slightly optimistic. Accuracy is at least 0.70 from layer 3 onward, though, so the result doesn't depend on that choice.
- **Small data:** 30 objects and 3 colors. Enough to see a clear effect, not to measure it precisely.
- **Representation only.** This shows the directions *exist*; it doesn't show the model *uses* them. The next step would be steering: adding the red direction to "The sky is…" and seeing whether the model says "red".

## Files

| file | what it does |
|---|---|
| `data.py` | Generates the dataset (templates → `data/*.json`) and runs sanity checks, e.g. no color word hidden in the test sentences ("blueberry" was caught and removed) |
| `data/*.json` | 120 explicit sentences, 120 implicit sentences, 48 control sentences |
| `color_directions.py` | The experiment: caches activations for every layer, computes directions, tests them. Runs cell by cell (`# %%`) in VS Code or as a script |

## Running it

Requires a GPU with about 6 GB of memory, plus `torch`, `transformer_lens`, `transformers` and `jaxtyping`.

```bash
python data.py              # optional: regenerate data/*.json
python color_directions.py  # ~2 min on an RTX 4060 laptop GPU
```

## Related work

- Mikolov et al. (2013), word2vec: word meanings as vector directions (`king − man + woman ≈ queen`)
- Abdou et al. (2021), *Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color*
- Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*: the difference-of-means method used here
- Park et al. (2023), *The Linear Representation Hypothesis and the Geometry of Large Language Models*

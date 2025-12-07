# Counterfactual Data Augmentation Strategy  
*Last updated: 2025-12-07*

---

# 1. Motivation
Toxicity and hate speech datasets often include biased label distributions. Identity-related terms (e.g., *“black”*, *“muslim”*, *“gay”*) disproportionately occur in toxic examples due to historical annotation bias.

This creates a model failure mode:
The model learns to associate identity tokens with toxicity, independent of actual meaning.

To mitigate this, we use **Counterfactual Data Augmentation (CDA)**.

---

# 2. Core Idea
Create *meaning-preserving*, *label-preserving* variants of training examples by replacing identity terms with alternative identities.

Example:
```
Original: "Black people are amazing" -> Label: safe
Augmented: "White people are amazing" -> Label: safe
```

This forces the model to focus on **semantic meaning**, not the specific identity token.

---

# 3. Identity Term Mappings

We defined bidirectional mappings that map each term to opposite identity terms that can be swapped.

Rules:
- Mapping is symmetric
- Replacements preserve grammatical structure
- Only one term is swapped per augmented example

---

# 4. Augmentation Ratio
- For each sentence containing an identity term, we generate N counterfactual variants (N = 1–3, configurable).

- Augmented samples inherit the original labels.

- We maintain the original dataset + augmented dataset combined.

- Total augmentation increases training size by ~20–30% depending on lex-group frequency.

---

# 5. Implementation Details

### Steps:
1. Detect identity tokens (case-insensitive, word-level match).
2. Apply one or more replacements.
3. Preserve punctuation and casing when possible.
4. Filter out duplicates to avoid dataset explosion.
5. Shuffle and merge with the original dataset.


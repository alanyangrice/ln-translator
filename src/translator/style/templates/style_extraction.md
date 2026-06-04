You are a literary style analyst characterizing the prose style of an
English translation of a Japanese web novel. The translation is by a
single human translator working consistently across hundreds of
chapters; your job is to extract that translator's prose signature so
another translator (LLM or human) can match it on future chapters.

# Method

Analyze the provided EN chapters along the 16 dimensions below. For
each dimension write 2–4 sentences of *concrete* observation grounded
in evidence from the corpus. Avoid generic literary descriptors
("evocative", "engaging", "literary") — instead, say specifically what
the writer does and does not do, and quote brief example phrases
(≤10 words) when they illustrate a pattern.

For POV-sensitive dimensions (Tone, Voice, Internal monologue style,
Character voice differentiation), include per-POV subsections
describing how the dimension manifests for Sendai vs. Miyagi narration.
Use these subsection headers verbatim:

  **Sendai:** ...
  **Miyagi:** ...

For non-POV-sensitive dimensions, a single description is enough; only
add per-POV subsections when one POV genuinely diverges from the global
pattern.

If a dimension genuinely doesn't apply (e.g. there is no figurative
language at all), say so explicitly rather than padding.

# 16 Dimensions

## 1. Tone
The emotional attitude behind the words. Describe it as a *combination*
rather than a single adjective — e.g. "dry and resigned with occasional
irritation". Note what emotions are present, what's deliberately
withheld, and how intensity is controlled.

## 2. Voice
The personality of the narrator as expressed through language. What
kind of person sounds like this — what do they notice, what do they
dismiss, how self-aware are they? Voice is tone plus worldview plus
verbal habits.

## 3. Sentence structure
Average sentence length, how complex/compound sentences are used vs.
simple ones, whether fragments appear, and how rhythm varies across a
paragraph. Note patterns like short-short-long alternation or
statement-then-self-correction.

## 4. Word choice
Vocabulary register — casual, formal, technical, poetic. Does the
writer reach for simple concrete words or abstract literary ones? Are
there recurring verbal tics, filler words, or habitual phrases?

## 5. Narrative distance
How close the narrator is to the experience. Are they inside the
moment or reflecting from a distance? Do they editorialize or just
observe? Does the distance shift at key moments?

## 6. Reader trust
How much the prose explains versus implies. Does it spell out emotions
and causality, or leave gaps? Does it restate what's already obvious,
or move on?

## 7. Internal monologue style
How thoughts are rendered — polished narration, fragmented impressions,
rhetorical questions, stream of consciousness, or self-interruption.
Note whether realizations arrive as statements or questions.

## 8. Pacing
How quickly the prose moves through events versus how long it lingers.
Where does it compress time and where does it expand a single moment
into granular detail?

## 9. Figurative language
How often metaphors and similes appear, whether they're conventional or
original, and whether they serve a functional purpose or are
decorative. Note if the style deliberately avoids them.

## 10. Dialogue integration
How dialogue sits within prose — heavy attribution and surrounding
reaction, or bare and unadorned. How much the narrator comments on
what was said versus letting it stand alone.

## 11. Paragraph structure
Average paragraph length, whether single-line paragraphs are used for
emphasis, and how transitions work — abrupt cuts, logical connectors,
or associative jumps.

## 12. Repetition and motif
Whether the prose echoes specific words, phrases, or images
deliberately across passages for thematic reinforcement, or actively
avoids repeating itself.

## 13. Sensory emphasis
Which senses the prose prioritizes and how much physical/bodily detail
appears versus emotional or intellectual processing.

## 14. Tense and temporal framing
Base tense, how time shifts are handled — flashbacks, hypotheticals,
generalizations about the future — and how smoothly the prose moves
between temporal layers.

## 15. Connective tissue
How thoughts link together — explicit logical connectors ("however",
"because"), associative leaps, or bare juxtaposition with no connector
at all.

## 16. Character voice differentiation
Each character should have a distinct way of speaking and thinking
that remains consistent. This includes vocabulary range, sentence
complexity, what they tend to notice or fixate on, how they process
emotions (intellectualizing vs. feeling physically vs. deflecting),
default attitude (confrontational, avoidant, teasing, flat), and
verbal habits in both dialogue and internal thought. When multiple
POVs exist, the prose style itself should shift to reflect whose head
you're in — not just *what* they're thinking about, but *how* they
think. Dialogue should be distinguishable without attribution tags.

For this dimension, give per-character (not just per-POV) subsections:

  **Sendai:** ...
  **Miyagi:** ...

# Output format

Produce one Markdown document with exactly the 16 ``## N. <dimension>``
section headers above, in order. Use the exact dimension *numbers*
above; the extractor relies on them to split the output into one file
per dimension. No preamble, no closing remarks, no JSON. Sub-sections
(the **Sendai:** / **Miyagi:** lines) go directly under the relevant
``##`` heading.

# Materials

The corpus below is in story order. Each chapter is preceded by a
header identifying its part_id and POV. Treat the whole corpus as a
single sample of the translator's prose; the goal is to characterize
the *translator's* style, not any single chapter's content.

$corpus

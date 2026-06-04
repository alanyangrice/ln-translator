# Style profile

The reference translator's prose signature, characterized along 16
dimensions. Each dimension lives in its own Markdown file in this
directory (`01-tone.md`, `02-voice.md`, …) so it can be hand-edited,
linked, or overridden independently. At prompt time the translator
concatenates all dimension bodies in numerical order and injects them
into the translation, comparison, and judge prompts via the
`$style_profile` placeholder.

## Bootstrap

```
translator style extract --through part_050
```

Reads the EN reference corpus through the named part (minus holdout
members), runs the extraction model, and writes one file per
dimension. Re-running overwrites all 16 files.

## Dimensions

1. Tone
2. Voice
3. Sentence structure
4. Word choice
5. Narrative distance
6. Reader trust
7. Internal monologue style
8. Pacing
9. Figurative language
10. Dialogue integration
11. Paragraph structure
12. Repetition and motif
13. Sensory emphasis
14. Tense and temporal framing
15. Connective tissue
16. Character voice differentiation

Until extraction has been run, the prompts will inject a placeholder
notice and translation continues normally.

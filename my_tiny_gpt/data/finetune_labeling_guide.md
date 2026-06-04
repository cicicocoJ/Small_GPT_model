# Fine-tuning Labeling Guide

## Task

`story_ending_polarity_classification`

Label the ending tendency of each short TinyStories text.

## Labels

`positive`: The ending is positive. A problem is solved, a character receives help, a conflict is softened, or the character ends safe, happy, or with a clear gain.

`negative`: The ending is negative. A problem remains unsolved, a character fails, feels lonely, afraid, or sad, a conflict is not repaired, or the bad result continues.

`discard`: The text is too short, too unclear, lacks an ending, has no obvious polarity, or is not suitable for the experiment.

## Principles

- Only use the information shown in the `text` field.
- Focus on the ending tendency, not just one emotional word.
- If there is a difficulty and it is solved at the end, label `positive`.
- If there is a difficulty and it is not solved at the end, label `negative`.
- If the text is neutral or impossible to judge, label `discard`.

## Examples

Positive:

> Tom lost his red ball. Mia found it under the bench and gave it back. They played together until dinner.

Negative:

> Tom lost his red ball. He looked under the bench but it was gone. He walked home with empty hands.

Discard:

> Tom had a ball. It was red.

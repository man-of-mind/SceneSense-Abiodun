# Pedestrian external-occlusion annotation rubric

## Independence and allowed evidence

Annotate each assigned panel independently. Use only the complete RGB view and
the two RGB crops in that panel. Do not discuss cases with the other annotator
until both CSV files are complete.

Do not consult perception outputs, coordinator-only manifests, depth or radar
data, semantic masks, diagnostic values, distances, or prior labels. The
diagnostic values used to balance the sample are deliberately hidden from the
annotation materials.

Do not change, add, remove, or reorder sample IDs in the CSV. Fill exactly one
visibility label and one truncation label for every row. Notes are optional.

## What counts as occlusion

Classify **external occlusion**: another object or scene surface blocks the
expected in-frame pedestrian body. Self-occlusion caused by body pose is not
occlusion. Image-boundary truncation is recorded separately and must not be used
as the external-occlusion label.

## Visibility labels

- `fully_visible`: essentially the entire in-frame pedestrian body is observable;
  there is no meaningful external occlusion.
- `partly_occluded`: approximately half or more of the expected in-frame body is
  observable.
- `largely_occluded`: some reliable person pixels are visible, but less than
  approximately half of the expected body.
- `not_observable`: no reliable pedestrian pixels are visually observable at the
  annotated position.
- `ambiguous`: image resolution, annotation alignment, or scene content prevents
  a defensible decision.

Use `ambiguous` only when the evidence does not support a defensible ordered
label; it is not a substitute for a difficult but decidable case.

## Truncation labels

- `none`: the expected body is not meaningfully cut by an image boundary.
- `partial`: an image boundary removes some, but not most, of the expected body.
- `severe`: an image boundary removes most of the expected body or makes its
  extent highly uncertain.

Truncation and external occlusion are independent fields. For example, a person
can be `fully_visible` for external occlusion and `partial` for truncation.

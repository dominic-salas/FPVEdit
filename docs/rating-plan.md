# Shot rating: criteria plan (proposed 2026-10-18, waiting for Dom's OK)

Goal (Dom): the rating decides which shots make a good montage. An indoor shot can beat an
outdoor one and the other way round; what counts is how clean and how impressive the trick
is. The rating should therefore replace the current "leave weak indoor shots out" rule.

What exists today: a 1-5 star rank of each shot's *strength* (trick weights plus speed per
square root of length) within the montage. It knows what was done, not how well.

## Criteria

Each shot gets a score from four parts, each 0-1, computed from signals the analysis already
measures (camera rotation per axis, motion-fit confidence, image speed, approach rate,
crash and ground cuts, brightness, noise, image-model scenery). Stars are then given within
the session, so the best shots of a quiet session still show up.

1. **Trick (what was done), weight 0.35**
   - Per trick: double flip or roll > flip, roll, power loop > yaw spin > close pass alone.
   - Combos: tricks linked within about 3 s with no settling in between count as one line
     and score more than the same tricks apart (Dom's "flow" labels).
   - Diminishing returns for the same trick repeated in one shot.

2. **Clean (how well), weight 0.35**
   - Completion: the turn reaches a full rotation for its type (measured angle in the range
     full turns measure at), not a half or a bail-out.
   - Recovery: after the trick the rotation settles to calm flight within about 1 s, with
     little wobble or oscillation (prop wash shows as fast back-and-forth rotation).
   - No impact: no crash, ground tap or sudden stop within 2 s after (the flight analysis
     already finds crashes; a tap shows as a sharp speed drop and a jolt).
   - Steady flow: speed is kept through the trick instead of stalling.

3. **Impressive (context), weight 0.2**
   - Proximity: a close pass or gap during or right around the trick (trick through or
     next to something) scores more than the same trick in open air.
   - Speed through the trick and the pass.
   - Scenery as a mild factor only (image model: open outdoor scenery a little up, cramped
     indoor a little down), never a filter on its own.

4. **Watchable (picture), weight 0.1**
   - Analog breakup, heavy noise or darkness during the trick, and motion blur so strong
     the trick cannot be followed, count against the shot.

Weights are a starting point and get tuned against the checks below, not by eye.

## How the montage uses it

- Shots are kept by rating instead of the indoor rule: everything 3 stars and up by default
  (a setting: "keep shots rated at least ..."), lower ones offered under Add a shot.
- The timeline colours and stars stay as built; the shot panel lists why a shot got its
  rating (for example "full back flip, clean recovery, through a gap").
- Variety: when many shots show the same trick, the weaker repeats lose a little.

## How it is checked (before anything ships)

Labelled data that already exists (Dom's hand labels and his finished edits):

- The 8 shots Dom marked **failed** (for example the flip that tapped the ground) must rate
  1-2 stars. Today's rank does not look at this at all.
- The 12 moments Dom **kept** in his own finished edits must rate above the 16 he **cut**
  more often than today's rank does (pairwise, kept vs cut in the same recording).
- Labelled flips, spins, weaves and the orbit must still be found (no recall lost).
- The indoor clip: its good shots can rate high, its dull ones low, instead of all being left
  out.

Weights are tuned on four of the labelled recordings and checked on the fifth (held out),
so the result is not just fitted to the same footage.

## Asking Dom (the best ground truth)

A quick way for Dom to give his own stars to shots in the review step (click the stars). The
app keeps them locally; 30-50 shots across a few sessions would show whether the criteria
agree with him and let the weights be tuned to his taste.

## Limits to keep in mind

- No model recognises trick quality directly; every criterion above is a measured stand-in
  (rotation, recovery, speed, proximity), so some ratings will disagree with Dom.
- Analog footage hides fast rotation in blur, which makes "completion" uncertain on the
  fastest tricks; those get a neutral cleanliness rather than a penalty.

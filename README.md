# Football Tactical Intelligence & Analysis Tool

A computer vision pipeline that takes raw football match footage and turns it into player tracking data, team assignments, match events, and eventually a player valuation, with each stage built and connected by hand rather than run through a single pre-built pipeline.

---

## What This Is

This project takes broadcast football footage and processes it through a sequence of stages, each one responsible for a single part of the problem: detecting players and the ball, tracking them across frames, mapping their positions onto real pitch coordinates, assigning them to teams, and identifying match events like possession changes and duels. The end goal is to take all of that frame-by-frame data and use it to infer formations and eventually produce a valuation for each player, based on what they actually did during the match rather than a single overall stat line.

---

## Architecture

```
Match Footage
      ↓
detection.py         (YOLOv9s)
      ↓
tracking.py          (ByteTrack / BoT-SORT)
      ↓
homography.py        (pixel space → pitch space)
      ↓
team_assignment.py   (HSV KMeans)
      ↓
event_detection.py   (possession / duel / transit)
      ↓
formation.py
      ↓
metrics.py
      ↓
valuation.py         (not yet started)
```

Every stage writes its output to a JSONL file keyed by frame number, which the next stage reads in. This was a deliberate choice made early on, explained more in [Design Decisions](#design-decisions).

---

## Pipeline Stages

### Detection
The detection stage uses a fine-tuned YOLOv9s model trained on the Roboflow `football-players-detection-3zvbc` dataset, with four classes: Ball, Player, Goalkeeper, and Referee. The current run (`train-7`) reached an mAP50 of about 0.836.

### Tracking
Once players are detected in each frame, ByteTrack/BoT-SORT assigns a persistent track ID to each one, so the same player keeps the same ID across frames even when they're briefly occluded or overlap with another player.

### Homography
The homography stage takes the pixel-space positions from tracking and maps them onto a flat, real-world pitch coordinate system, using `cv2.findHomography` with RANSAC on detected pitch keypoints. This is the stage giving the most trouble right now; around 65% of frames are currently being skipped due to low-confidence keypoint detection, and I haven't fully resolved this yet. More on this in [Design Decisions](#the-homography-problem-still-unresolved).

### Team Assignment
Rather than using an embedding-based approach like SigLIP+UMAP, team assignment is done by cropping the torso region of each detected player, converting to HSV, and clustering on Hue and Saturation with KMeans. The model is fit once on a sample of crops and then used to predict team labels for every subsequent frame. This was a case of not overcomplicating a problem that didn't need a heavy solution; for instance, telling two jersey colors apart doesn't require an embedding model.

### Event Detection
This stage looks at tracked positions frame to frame and classifies events like possession, duels, and ball transit using a Bayesian classification approach. Real frame numbers are attached at every step here rather than an incrementing counter, which matters because of the frame skipping happening upstream in homography.

### Formation
For each possession window from event detection, `formation.py` averages every player's on-pitch position over that window (separately for the possessing and defending team), then matches those average positions against a bank of 12 template formations (4-4-2, 4-3-3, 3-5-2, etc.) using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to find the lowest-cost assignment between observed players and template slots. The formation template with the smallest summed distance is recorded as the team's shape for that window, alongside the distance itself as a rough confidence score.

This stage is functional but its accuracy is capped by what's upstream: ID fragmentation from tracking/homography noise (see below) means "average position" windows are sometimes built from more track IDs than there are actual players on the pitch, which inflates the distance score and occasionally misclassifies short or noisy windows.

### Metrics
`metrics.py` takes the same possession windows and, for every track ID active in a window, computes three per-player statistics: total distance covered (with implausible frame-to-frame jumps filtered out as tracking noise), sprint count (number of times a player's speed crosses a threshold, using a lookahead over frames rather than raw frame-to-frame speed to smooth out jitter), and zone occupancy (a 3x3 grid over the pitch, tallying how many frames the player spent in each zone). Output is written per window as JSONL, matching the format of every other stage.

Because track IDs aren't yet resolved to a single physical player across ID switches, these per-window stats should be read as "this track ID's activity," not yet "this player's activity" — that reconciliation hasn't been built.

### Valuation
Not started yet. The plan is to take the formation and metrics output and turn it into a final per-player valuation.

---

## Results

| Metric | Value |
|---|:---:|
| Detection mAP50 (train-7) | 0.836 |
| Homography frame skip rate | ~65% (unresolved) |
| Possession windows detected (~10 min clip) | 15 |
| Distinct track IDs in a single window | up to 76 (should be ≤22) |

> Numbers above are from a single ~10-minute clip (Chelsea vs. Burnley, 2015-02-21), not a full match — see `rebuild_files.py`. The track ID count row is the clearest symptom of the ID-fragmentation issue described in [Formation](#formation) and [Metrics](#metrics). Valuation numbers will be added once that stage exists.

---

## Design Decisions

Several decisions had to be made across the pipeline, ranging from how data gets passed between stages to how team assignment should actually work under the hood.

### Choosing JSONL Over JSON/CSV
Each stage writes streamed, nested, variable-length data per frame, and JSONL just fits that better. A single JSON file has to be loaded whole and rewritten every time something new gets appended, which starts to hurt once you're working with a full match instead of a handful of test frames. JSONL avoids that; each frame gets appended and read back on its own. As a side effect, this also made debugging a lot easier, since I could inspect one stage's output directly instead of having to re-run the whole pipeline to check if a fix actually worked.

### Real Frame Numbers, Not a Counter
Homography currently skips a large chunk of frames (more on that below), and this ended up being the reason keying everything to the real frame number mattered so much. An incrementing counter would have quietly desynced every downstream stage the moment frames started getting dropped; event detection would still run fine, it just would have been reasoning about the wrong moment in the match without any error being thrown. Using the actual frame number instead means a skip just shows up as a gap in the data, not a silent shift in everything after it.

### Team Assignment: HSV KMeans
The first instinct here was to reach for an embedding-based approach, something like SigLIP+UMAP, since that felt like the "correct" way to do clustering. But separating two jersey colors doesn't need that much. Cropping the torso region, converting to HSV, and clustering on Hue and Saturation was enough signal, fit once and then predicted for the rest of the match. The harder and more interesting part of this project is further downstream; I'd rather spend the complexity budget on event detection and valuation than on telling two shirt colors apart.

### Path Anchoring
`Path(__file__).resolve()` gets used everywhere instead of relying on the working directory. Running the same script from a terminal versus PyCharm's run button resolves paths differently, and I hit that bug enough times early on that anchoring to the script's own location just made it go away for good instead of patching it per script.

### The Homography Problem — Still Unresolved
This is the one I don't have a clean answer for yet. About 65% of frames are being skipped because the keypoint detection model isn't confident enough about the pitch lines in those frames, so `homography.py` drops them instead of producing a bad transform. I'm leaving this documented as unresolved rather than pretending it's fixed; the likely path forward is either a stronger keypoint model or interpolating between frames that were detected confidently, but I haven't committed to either yet.

This is also the root cause of the ID fragmentation showing up in formation and metrics output: with two-thirds of frames dropped, ByteTrack loses and re-acquires players constantly, so a single physical player ends up scattered across many track IDs. Formation and metrics compute correctly given their inputs, but until tracking is more continuous (or track IDs get reconciled downstream), neither stage's numbers should be read as "per-player" yet — see [Formation](#formation) and [Metrics](#metrics).

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run each stage in order:

```bash
python src/detection.py --video path/to/match.mkv
python src/tracking.py
python src/homography.py
python src/team_assignment.py
python src/event_detection.py
python src/formation.py
python src/metrics.py
```

Each stage reads its input from the JSONL file the previous stage wrote to `data/`, so they need to be run in order the first time through. `src/rebuild_files.py` runs the whole chain (tracking → keypoints → team assignment → event detection → formation → metrics) against a shared frame count in one script, which is useful when a change upstream means everything downstream needs regenerating.

---

## What I Learned

Breaking the pipeline into stages that each read and write their own JSONL file turned out to be one of the better decisions I made early on, mostly because of how it helped with debugging. Two bugs stand out. One was a capitalization mismatch between `"Referee"` and `"referee"` that silently emptied out the fitting sample used for team assignment; nothing crashed, it just quietly produced garbage. The other was pointing team assignment at the pose model's weights instead of the detection model's, which is an easy mistake to make when both are just file paths sitting a few directories apart. Both were things I only caught by inspecting a single stage's output on its own, rather than trying to trace the bug through the entire pipeline at once.

The homography skip rate is still the most annoying open problem. It's the kind of bug that's obvious the moment you look at it and genuinely hard to fix properly. If anything it's been a good reminder that a model "mostly working" and a pipeline "working" aren't the same claim once you're chaining several stages together and errors get a chance to compound.

Building `metrics.py` surfaced two more bugs in the same category — quiet rather than crashing. The sprint counter was checking `frame_num` against the set of known frames instead of `frame_num + 4` (the frame it was actually about to look up), so it always found a match and never skipped the lookahead when it should have. And zone occupancy was keyed by `(row, col)` tuples, which `json.dumps` can't serialize as dict keys — that one would have thrown on the very first write, so it's really a "didn't discover it until I ran the whole thing end to end" bug rather than a silent one. Both are fixed now, alongside clipping zone fractions to `[0, 1]` so a position slightly outside the pitch bounds doesn't produce an out-of-range grid index.

---

## References

- [Roboflow `football-players-detection-3zvbc` dataset](https://universe.roboflow.com/)
- [`roboflow/sports`](https://github.com/roboflow/sports) — used as a structural reference for the pitch keypoint scheme and `SoccerPitchConfiguration`
- [SoccerNet](https://www.soccer-net.org/)

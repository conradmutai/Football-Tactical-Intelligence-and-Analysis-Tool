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
valuation.py
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

### Formation, Metrics, and Valuation
These three stages are still in progress. The plan is for `formation.py` to infer formation shape and player roles from the tracked positions, `metrics.py` to aggregate per-player statistics across the match, and `valuation.py` to turn all of that into a final per-player valuation.

---

## Results

| Metric | Value |
|---|:---:|
| Detection mAP50 (train-7) | 0.836 |
| Homography frame skip rate | ~65% (unresolved) |

> Formation, metrics, and valuation numbers will be added once those stages are actually working.

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
```

Each stage reads its input from the JSONL file the previous stage wrote to `data/`, so they need to be run in order the first time through.

---

## What I Learned

Breaking the pipeline into stages that each read and write their own JSONL file turned out to be one of the better decisions I made early on, mostly because of how it helped with debugging. Two bugs stand out. One was a capitalization mismatch between `"Referee"` and `"referee"` that silently emptied out the fitting sample used for team assignment; nothing crashed, it just quietly produced garbage. The other was pointing team assignment at the pose model's weights instead of the detection model's, which is an easy mistake to make when both are just file paths sitting a few directories apart. Both were things I only caught by inspecting a single stage's output on its own, rather than trying to trace the bug through the entire pipeline at once.

The homography skip rate is still the most annoying open problem. It's the kind of bug that's obvious the moment you look at it and genuinely hard to fix properly. If anything it's been a good reminder that a model "mostly working" and a pipeline "working" aren't the same claim once you're chaining several stages together and errors get a chance to compound.

---

## References

- [Roboflow `football-players-detection-3zvbc` dataset](https://universe.roboflow.com/)
- [`roboflow/sports`](https://github.com/roboflow/sports) — used as a structural reference for the pitch keypoint scheme and `SoccerPitchConfiguration`
- [SoccerNet](https://www.soccer-net.org/)

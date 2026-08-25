# Soccer Vision — Player Tracking & Valuation Pipeline

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
The homography stage takes the pixel-space positions from tracking and maps them onto a flat, real-world pitch coordinate system, using `cv2.findHomography` with RANSAC on detected pitch keypoints. This is the stage giving the most trouble right now — around 65% of frames are currently being skipped due to low-confidence keypoint detection, and I haven't fully resolved this yet. More on this in [Design Decisions](#the-homography-problem-still-unresolved).

### Team Assignment
Rather than using an embedding-based approach like SigLIP+UMAP, team assignment is done by cropping the torso region of each detected player, converting to HSV, and clustering on Hue and Saturation with KMeans. The model is fit once on a sample of crops and then used to predict team labels for every subsequent frame. This was a case of not overcomplicating a problem that didn't need a heavy solution — telling two jersey colors apart doesn't require an embedding model.

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

**JSONL for everything.** Every stage in this pipeline writes streamed, nested, variable-length data on a per-frame basis, and JSONL fits that a lot better than a single JSON file or a CSV would. With JSON, the whole file has to be loaded into memory and rewritten every time something is appended, which becomes a problem once you're processing an entire match rather than a handful of frames. JSONL lets each frame get appended and read back independently, which also made it a lot easier to debug one stage at a time instead of running the whole pipeline just to check if one part worked.

**Frame numbers instead of a counter.** Since homography ends up skipping close to 65% of frames, keying everything downstream to the actual frame number instead of a running index that increments every time a frame is processed turned out to matter a lot. If I had used an incrementing counter, the moment homography started skipping frames, everything downstream would have drifted out of sync with what was actually happening in the video, and event detection would have been reasoning about the wrong point in time without any obvious sign that something was wrong. Keying to the real frame number means a skipped frame is just a gap, not a silent shift in everything after it.

**Team assignment via HSV KMeans.** The first instinct was to reach for something like SigLIP+UMAP, since that's the more "correct" way of doing this kind of clustering. But separating two jersey colors doesn't really need that. Cropping the torso and clustering on HSV Hue and Saturation was enough signal to do the job, and it kept the pipeline lighter without pulling in a heavy embedding model for a problem that didn't need one. The harder, more interesting part of this project is further downstream in event detection and valuation, not in telling two shirt colors apart.

**Path handling.** Every script anchors its paths using `Path(__file__).resolve()` rather than relying on the current working directory. Running the same script from a terminal versus from PyCharm's run button resolves differently otherwise, and this was a bug I kept running into early on until I just anchored everything to the script's own location instead of working around it each time.

**The homography problem (still unresolved).** This is the part of the pipeline I'm least happy with right now. Around 65% of frames get skipped because the keypoint detection model isn't confident enough about where the pitch lines are in those frames, which means `homography.py` just drops them rather than producing a bad transform. I noted this here instead of pretending it's solved, because it isn't yet — the likely fix is either a stronger keypoint model or some kind of interpolation between frames where keypoints were confidently detected, but I haven't committed to either approach yet.

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

Splitting the pipeline into stages that each read and write their own JSONL file made debugging a lot more manageable than it would have been as one long script. A couple of real bugs got caught this way — a capitalization mismatch between `"Referee"` and `"referee"` that quietly emptied out the fitting sample for team assignment, and a case where I was pointing at the pose model's weights instead of the detection model's weights without immediately realizing it, since both are just paths. Being able to inspect one stage's output on its own made both of these obvious pretty quickly, instead of having to debug the whole pipeline at once to find them.

The homography skip rate has been the most frustrating problem so far, mostly because it's easy to notice and hard to actually fix. It's a good reminder that "the model detects things most of the time" and "the pipeline works" aren't the same statement once errors compound across stages.

---

## References

- [Roboflow `football-players-detection-3zvbc` dataset](https://universe.roboflow.com/)
- [`roboflow/sports`](https://github.com/roboflow/sports) — used as a structural reference for the pitch keypoint scheme and `SoccerPitchConfiguration`
- [SoccerNet](https://www.soccer-net.org/)

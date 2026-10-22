# Football Tactical Intelligence & Analysis Tool

A computer vision pipeline that takes raw football match footage and turns it into player tracking data, team assignments, match events, formation classification, and per player metrics, with each stage built and connected by hand rather than run through a single pre built pipeline.

---

## What This Is

This project takes broadcast football footage and processes it through a sequence of stages, each one responsible for a single part of the problem: detecting players and the ball, tracking them across frames, mapping their positions onto real pitch coordinates, assigning them to teams, identifying match events like possession changes and duels, classifying each team's formation, and computing per player statistics across the match.

This was originally scoped to end in a player valuation stage as well, turning everything upstream into a single number per player. I decided against building it. More on why in [Design Decisions](#dropping-valuation).

---

## Architecture

```
Match Footage
      ↓
detection.py         (YOLOv9s)
      ↓
tracking.py          (ByteTrack / BoT-SORT)
      ↓
homography.py        (pixel space to pitch space)
      ↓
team_assignment.py   (HSV KMeans)
      ↓
event_detection.py   (possession / duel / transit)
      ↓
formation.py
      ↓
metrics.py
```

Every stage writes its output to a JSONL file keyed by frame number, which the next stage reads in. This was a deliberate choice made early on, explained more in [Design Decisions](#design-decisions).

---

## Pipeline Stages

### Detection
People and the ball are detected with a COCO pretrained YOLOv9m (`src/yolov9m.pt`), using the `person` and `sports ball` classes. The fine tuned YOLOv9s model trained on the Roboflow `football-players-detection-3zvbc` dataset (`train-7`, mAP50 about 0.836) is still in the pipeline, but only as a *role tagger*: it runs every fifth frame and its Goalkeeper and Referee boxes are matched onto the person boxes by overlap, then voted per track.

That split is the opposite of how the project started, and it is worth explaining. Measured over frames 500-1500, the custom detector emitted about 30 boxes per frame for roughly 17 real people. The extras were not distant players, they were body fragments: a full body box plus a separate torso or legs box for the same person, which NMS does not suppress because a torso box inside a body box only has an IoU of about 0.4. Those fragments flickered on and off, and every flicker spawned a track. Checking frame by frame, *every* box the custom detector found overlapped a COCO box, so nothing was lost by the swap. See [Results](#results) for the before and after.

### Tracking
Detections are de-duplicated and then handed to BoT-SORT (`config/tracker.yaml`), which assigns a persistent track ID to each person. Two details matter:

The tracker is driven directly rather than through Ultralytics' `model.track()`. That convenience wrapper associates on the *raw* detections, so duplicate boxes reach the tracker before anything can remove them, which was the whole problem. Driving `BOTSORT.update()` by hand means the de-duplication happens first, and lets the ball skip tracking entirely since it needs a position, not an identity.

`gmc_method: sparseOptFlow` is on, because plain IoU association fails on a panning broadcast camera: the pan alone moves players far enough between frames to push IoU under the match threshold and kill the track.

### Homography
The homography stage takes the pixel space positions from tracking and maps them onto a flat, real world pitch coordinate system, using `cv2.findHomography` with RANSAC on detected pitch keypoints. This is the stage giving the most trouble right now. On a full ~10 minute run (15,549 frames), about 48.6% of frames are being skipped due to low confidence keypoint detection, and I haven't fully resolved this yet. More on this in [Design Decisions](#the-homography-problem-still-unresolved).

### Team Assignment
Rather than using an embedding based approach like SigLIP+UMAP, team assignment is done by cropping the torso region of each detected player, converting to HSV, and clustering on Hue and Saturation with KMeans. The model is fit once on a sample of crops and then used to predict team labels for every subsequent frame. This was a case of not overcomplicating a problem that didn't need a heavy solution; telling two jersey colors apart doesn't require an embedding model.

### Event Detection
This stage looks at tracked positions frame to frame and figures out who has the ball, using a nearest player distance rule rather than anything more elaborate. For each frame, the player closest to the ball gets checked against two radii, a tight one for clean possession and a wider one for a contested duel. Anything beyond both radii is marked as nobody in possession. Consecutive frames with the same team in possession collapse into a single possession window, with a minimum duration filter so a one off frame of noise doesn't register as a real change. Real frame numbers are attached at every step here rather than an incrementing counter, which matters because of the frame skipping happening upstream in homography.

### Formation
For each possession window from event detection, `formation.py` averages every player's on pitch position over that window (separately for the possessing and defending team), then matches those average positions against a bank of 12 template formations (4-4-2, 4-3-3, 3-5-2, etc.) using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to find the lowest cost assignment between observed players and template slots. The formation template with the smallest summed distance is recorded as the team's shape for that window, alongside the distance itself as a rough confidence score.

These 12 templates are hand authored, not learned. The research this stage is based on, Shaw and Glickman's formation analysis paper, discovers its templates by clustering tracking data across roughly 100 matches. I have one match, so running the same clustering step here would realistically only ever discover a handful of shapes and call it equivalent to the paper's approach, which it isn't. Hand authoring 12 commonly used modern formations instead trades "the templates came from data" for a result that's actually legible: a label like "4-3-3" is something a reader can sanity check directly against the footage, in a way a raw cluster centroid isn't.

This stage is functional but its accuracy is capped by what's upstream. ID fragmentation from tracking and homography noise (see below) means "average position" windows are sometimes built from more track IDs than there are actual players on the pitch, which inflates the distance score and occasionally misclassifies short or noisy windows.

### Metrics
`metrics.py` takes the same possession windows and, for every track ID active in a window, computes three per player statistics: total distance covered (with implausible frame to frame jumps filtered out as tracking noise), sprint count (number of times a player's speed crosses a threshold, using a lookahead over frames rather than raw frame to frame speed to smooth out jitter), and zone occupancy (a 3x3 grid over the pitch, tallying how many frames the player spent in each zone). Output is written per window as JSONL, matching the format of every other stage.

Because track IDs aren't yet resolved to a single physical player across ID switches, these per window stats should be read as "this track ID's activity," not yet "this player's activity." That reconciliation hasn't been built.

---

## Results

All figures from the same ~10-minute clip (Chelsea vs. Burnley, 2015-02-21, 15,000 frames at 25 fps).

### Tracking, before and after the detector swap

| Metric | Custom detector + ByteTrack | COCO detector + BoT-SORT |
|---|:---:|:---:|
| Distinct track IDs | 5,553 | **1,360** |
| Median track lifetime | 0.4 s | **3.0 s** |
| Tracks lasting <= 5 frames | 2,141 | **109** |
| Tracks lasting > 250 frames | 56 | **166** |
| Ball detected | 17% of frames | **37%** |
| Next-frame box match (frames 500-1500) | 0.856 | **0.986** |

1,360 IDs is still well above the ~25 people on the pitch, and that is expected rather than a remaining bug: a ten minute broadcast clip contains replays, close-ups and dozens of camera cuts, and every cut legitimately ends a track. Over a continuous 1,000-frame passage of play the same pipeline produces **83** IDs with a 5.9 s median lifetime. `player_identity.py` is what reconnects tracks across those cuts, where the shirt number is legible.

### Geometry and events

| Metric | Before | After |
|---|:---:|:---:|
| Homography median reprojection error | 3.3 m | **0.02 m** |
| Frames with a validated homography | ~53% (unvalidated) | ~24-32% (validated) |
| Possession windows (30 s clip) | 1 | **20** |
| Possession windows (10 min clip) | 15 | **162** |

The homography coverage going *down* is the point: the frames that were dropped were producing transforms with a median error of 3.3 m, and 29% of them were worse than 10 m. Those were what made pitch-space player speeds come out at a median of 83 m/s.

### What is still weak

- **Distances are conservative.** Only about a quarter of frames carry valid geometry, so gaps in a trajectory are skipped, and steps implying more than 12 m/s are rejected as jitter. The numbers understate rather than overstate.
- **Jersey numbers resolve for a minority of players.** OCR succeeds on about a third of the crops worth attempting, and only large, rear-facing players are ever worth attempting.
- **The pitch keypoint model is the remaining weak link**, and fixing it properly means retraining it, not tuning around it.

---

## Design Decisions

Several decisions had to be made across the pipeline, ranging from how data gets passed between stages to how team assignment should actually work under the hood.

### Choosing JSONL Over JSON/CSV
Each stage writes streamed, nested, variable length data per frame, and JSONL just fits that better. A single JSON file has to be loaded whole and rewritten every time something new gets appended, which starts to hurt once you're working with a full match instead of a handful of test frames. JSONL avoids that; each frame gets appended and read back on its own. As a side effect, this also made debugging a lot easier, since I could inspect one stage's output directly instead of having to re run the whole pipeline to check if a fix actually worked.

### Real Frame Numbers, Not a Counter
Homography currently skips a large chunk of frames (more on that below), and this ended up being the reason keying everything to the real frame number mattered so much. An incrementing counter would have quietly desynced every downstream stage the moment frames started getting dropped; event detection would still run fine, it just would have been reasoning about the wrong moment in the match without any error being thrown. Using the actual frame number instead means a skip just shows up as a gap in the data, not a silent shift in everything after it.

### Team Assignment: HSV KMeans
The first instinct here was to reach for an embedding based approach, something like SigLIP+UMAP, since that felt like the "correct" way to do clustering. But separating two jersey colors doesn't need that much. Cropping the torso region, converting to HSV, and clustering on Hue and Saturation was enough signal, fit once and then predicted for the rest of the match. The harder and more interesting part of this project is further downstream; I'd rather spend the complexity budget on event detection and formation classification than on telling two shirt colors apart.

### Formation Templates: Hand Authored, Not Clustered
Covered above in [Formation](#formation), but worth repeating here as its own decision: the templates driving formation classification are 12 fixed, commonly used modern formations rather than templates discovered through clustering. With one match instead of the roughly 100 the reference paper uses, clustering my own data wouldn't have produced anything close to equivalent, so hand authoring a recognizable, defensible set of templates was the more honest choice.

### Dropping Valuation
This project was originally scoped to end in a player valuation stage, turning formation and metrics output into a single number per player. I decided against building it. Valuation needs some model of what actually makes a player valuable, calibrated against real market or performance data, and I don't have that data or a principled way to build the model without it. Rather than hand wave a formula together, I chose to keep the project scoped to what I can actually verify end to end: detection, tracking, tactical shape, and per player physical output. If I revisit valuation later, it belongs as its own project built on top of this one, not a rushed final stage bolted onto it.

### Path Anchoring
`Path(__file__).resolve()` gets used everywhere instead of relying on the working directory. Running the same script from a terminal versus PyCharm's run button resolves paths differently, and I hit that bug enough times early on that anchoring to the script's own location just made it go away for good instead of patching it per script.

### The Homography Problem, Improved But Still The Weak Link
This is still the least solid stage, but it is no longer the thing breaking everything downstream.

The original failure was not that frames were being skipped, it was that the frames being *kept* were wrong. The keypoint model emits all 32 pitch keypoints for every frame whether or not the landmark is visible, and the low confidence ones land on empty grass. Fitting a homography to those produced a median reprojection error of 3.3 m, with 29% of frames worse than 10 m. That is what made pitch space player speeds come out at a median of 83 m/s.

Three changes, each validated by measurement rather than taste:

- **Filter the keypoints before fitting.** Only keypoints at confidence >= 0.7 are used, and at least 6 are required. Four would fit any homography exactly and prove nothing.
- **Validate the fit.** `ViewTransformer` now keeps RANSAC's inlier mask and its own reprojection error, and `build_transformer_for_frame` throws out any frame whose fit disagrees with the landmarks it was built from.
- **Check the fit is well conditioned.** A fit can match its own landmarks perfectly and still be useless if they sit in one small corner of the image, because it then extrapolates nonsense out to where the players are. The keypoints must span real image area, and the video frame must map to a convex, plausibly sized patch of pitch.

Median reprojection error went from 3.3 m to 0.02 m, at the cost of coverage falling from about 53% of frames to about 32%. That trade is deliberate: downstream stages already skip missing frames, and a missing radar is better than a wrong one.

What remains is that the surviving homographies still jitter frame to frame, because the underlying keypoint model is weak. `metrics.py` compensates by median filtering each trajectory over a short window and rejecting steps faster than 12 m/s, so distances come out conservative rather than absurd. The real fix is a better keypoint model, which is a retraining job, not a tuning one.

### Possession Is Measured In Image Space
Possession asks who is nearest the ball, and that does not need a pitch model. Tying it to one was costing most of the data: the homography resolves on about 32% of frames while the ball is seen on about 64%, so pitch space possession could only ever see their 19% intersection, which was not enough to form possession windows at all on a short clip.

It now measures in image space, using the median player bounding box height as a ruler to convert the real world possession and duel radii into pixels at the current camera zoom. Over the same 30 second clip this took possession from 1 window to 20.

### Shirt Colour Gives The Team, The Number Gives The Player
These are often spoken about together as a way to track players, but they do very different jobs.

Shirt colour can only ever give the team. Eleven team-mates wear the same kit, so colour is a gate that stops cross-team merges, never an identity.

The jersey number is the only real identity signal, and it is sparse. A median player box is about 60 px tall, which puts the number at roughly 11 px, well under the ~16-20 px OCR needs. Only large, rear facing players are readable at all, and nothing below a 45 px box ever is.

What makes it usable anyway is voting across a whole tracklet instead of trusting any single frame, and that only became possible after the detector swap: at a 0.3 s median tracklet a player got no legible reads at all, while at ~6 s there are dozens of chances. `player_identity.py` reads the clearest crops per tracklet, folds partial reads (OCR often catches only the first digit of a two digit number) into the full read, and accepts a number when the vote is consistent. Tracklets carrying the same number and team, that never overlap in time, are merged into one player. Tracklets with no legible number keep a synthetic id rather than being merged on a guess.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Everything runs over an explicit frame range, so you can analyse a 30 second
clip or a whole half with the same commands. Ranges accept frames (`1500`),
seconds (`90s`), minutes (`10m`) or a clock (`2:30`).

```bash
# 1. the analysis pipeline: tracking -> keypoints -> teams -> events -> formation -> metrics
python -m src.rebuild_files --video-id chelsea_burnley --start 0 --duration 10m

# 2. resolve tracker ids into players, reading shirt numbers where they are legible
python -m src.player_identity --video-id chelsea_burnley --start 0 --duration 10m

# 3. render the annotated video with the live stats panel
python -m src.render_video --video-id chelsea_burnley --start 0 --duration 10m \
    --team-names "Chelsea,Burnley"
```

Every stage writes `data/<stage>_output/<video-id>_<start>-<end>.jsonl`, with the
range in the filename so separate runs never overwrite each other. Frame numbers
inside those files are always absolute, so a run starting at `5m` joins correctly
against anything else taken from the same video.

Step 2 is optional. Without it the overlay falls back to raw track ids. Step 3
needs step 1 to have been run over the same range.

The two colour clusters come out of KMeans in an arbitrary order, so if the team
names land on the wrong kits, re-run step 3 with `--swap-teams`. The renderer
prints which cluster got which colour, and the stats panel shows a colour swatch
beside each team name.

---

## What I Learned

Breaking the pipeline into stages that each read and write their own JSONL file turned out to be one of the better decisions I made early on, mostly because of how it helped with debugging. Two bugs stand out. One was a capitalization mismatch between `"Referee"` and `"referee"` that silently emptied out the fitting sample used for team assignment; nothing crashed, it just quietly produced garbage. The other was pointing team assignment at the pose model's weights instead of the detection model's, which is an easy mistake to make when both are just file paths sitting a few directories apart. Both were things I only caught by inspecting a single stage's output on its own, rather than trying to trace the bug through the entire pipeline at once.

The homography skip rate is still the most annoying open problem. It's the kind of bug that's obvious the moment you look at it and genuinely hard to fix properly. If anything it's been a good reminder that a model "mostly working" and a pipeline "working" aren't the same claim once you're chaining several stages together and errors get a chance to compound.

Building `metrics.py` surfaced two more bugs in the same category: quiet rather than crashing. The sprint counter was checking `frame_num` against the set of known frames instead of `frame_num + 4` (the frame it was actually about to look up), so it always found a match and never skipped the lookahead when it should have. And zone occupancy was keyed by `(row, col)` tuples, which `json.dumps` can't serialize as dict keys; that one would have thrown on the very first write, so it's really a "didn't discover it until I ran the whole thing end to end" bug rather than a silent one. Both are fixed now, alongside clipping zone fractions to `[0, 1]` so a position slightly outside the pitch bounds doesn't produce an out-of-range grid index.

---

## References

- [Roboflow `football-players-detection-3zvbc` dataset](https://universe.roboflow.com/)
- [`roboflow/sports`](https://github.com/roboflow/sports), used as a structural reference for the pitch keypoint scheme and `SoccerPitchConfiguration`
- [SoccerNet](https://www.soccer-net.org/)
- Shaw, L. and Glickman, M., "Inferring and Evaluating Team Strategy in Football Using Automated Formation Analysis of Tracking Data," the reference for the formation classification approach this stage is a simplified version of
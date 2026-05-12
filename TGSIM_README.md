# Automated PII Redaction for Public Media Sharing  
## EgoBlur Gen2: Method, Pipeline, and Limitations

**Audience:** Teams publishing images or video to the public who need a clear explanation of *how* personally identifiable information (PII) is reduced before release.  
**Scope:** Face and license-plate handling using Meta’s open **EgoBlur** Gen2 models and the `egoblur-gen2` tooling in this repository.

---

### 1. Purpose and privacy goal

When imagery is shared externally—websites, datasets, press material, or partner deliverables—two common sources of PII are **human faces** and **vehicle license plates**. Even when other metadata is stripped, visible faces and readable plates can identify individuals or tie vehicles to people and locations.

The workflow described here is **detect-then-redact**: software finds regions likely to contain faces or plates, then **permanently overwrites** those pixels in the exported image or video so they are no longer recognizable in the distributed file. This is a **risk-reduction** measure: it lowers the chance that PII appears in public copies. It is **not** a formal cryptographic guarantee that every pixel of every identity is removed, because any detector can miss objects or occasionally misfire (see Section 5).

---

### 2. What algorithm and models are used?

**Product name:** EgoBlur (Gen2 in our setup), from Meta’s Project Aria open tooling—documented publicly and distributed as TorchScript (`.jit`) detector weights together with reference code.

**Core idea:** Two **separate deep-learning object detectors**, each shipped as a **TorchScript** model and loaded with **PyTorch**:

1. **Face detector** — predicts axis-aligned **bounding boxes** and **confidence scores** for face instances.  
2. **License-plate detector** — same structure, trained for plates.

Internally, inference follows a **Detectron2-style** pattern (the codebase vendors Detectron2 utilities for scripted models): the model produces **instance predictions** (boxes, class labels, scores). Outputs are filtered to the target class (face **or** plate), then **non-maximum suppression (NMS)** merges overlapping boxes using an IoU threshold. A **score threshold** (between 0 and 1) drops low-confidence detections. Optional **camera-specific default thresholds** exist for Aria-style streams; for general RGB imagery, operators can override with explicit thresholds (as we did when lowering thresholds to catch more marginal detections).

**Redaction primitive:** After all accepted boxes are collected, the tool does **not** store names, embeddings, or biometric templates. It only uses geometry: for each box, a **strong blur** is applied inside the region (via OpenCV), and results are **composited** back into the frame using an **elliptical mask** aligned to the box so the transition respects the approximate shape of a face or plate. Bounding boxes can be **scaled** (e.g. slightly enlarged) so blur coverage is conservative around the detection.

**Hardware:** Models run on **CPU or CUDA** depending on the machine. Public-facing batch jobs benefit from GPU where available; Mac deployments often use CPU (slower but functionally equivalent for correctness of the method).

---

### 3. What we are doing in this project (operational pipeline)

1. **Ingest** source images or video frames (PNG, JPEG, or MP4 via the same CLI).  
2. **Load** `ego_blur_face_gen2.jit` and `ego_blur_lp_gen2.jit` from disk.  
3. **Run** face and plate detectors on each frame (or still image once).  
4. **Tune** sensitivity using `--face_model_score_threshold` and `--lp_model_score_threshold` (lower = keep weaker detections, more blur but more false positives) and optionally `--scale_factor_detections` to widen blur regions.  
5. **Emit** a new image or video file where sensitive regions are blurred—**only this output** is intended for public sharing; originals stay internal per your data policy.

For **video**, the same per-frame detection and blur loop applies; audio and non-visual tracks are unchanged unless you use separate tooling. For **VRS** (Aria Gen2 native recordings), a dedicated `egoblur-vrs-blur` path exists; the principle remains detect-and-blur on camera streams.

**Video-specific note:** Each decoded frame is treated as an independent still for detection (temporal smoothing across frames is not a built-in feature of the demo CLI). Fast motion can occasionally produce frames where a face or plate is harder to detect; for long-form public releases, spot-check high-motion segments or consider slightly more aggressive thresholds for those clips.

**NMS (`--nms_iou_threshold`):** After the neural network proposes overlapping boxes, **non-maximum suppression** keeps the strongest box when candidates overlap heavily. Adjusting this IoU threshold changes how aggressively duplicates are merged; it complements score thresholds when many overlapping proposals appear.

---

### 4. How this supports “no PII” narratives for public release

Stakeholders should understand three layers:

| Layer | What it means |
|--------|----------------|
| **Technical** | PII *surfaces* (faces, plates) are targeted by learned detectors and **pixel-level blur** is applied before export. |
| **Process** | Raw captures remain controlled; a **redacted derivative** is produced for publication. Access and retention policies still apply to originals. |
| **Residual risk** | Missed detections, partial occlusion, motion blur, extreme angles, or artistic content may leave readable PII. **Human spot-checks** on sampled frames and **threshold tuning** are recommended for high-stakes releases. |

This approach aligns with common **privacy-by-design** practice for media: minimize identifiable content in the artifact that leaves the trust boundary, without claiming impossible perfection.

**Suggested operational checklist before wide publication**

1. Decide whether **faces**, **plates**, or **both** must be redacted for the audience and jurisdiction.  
2. Run the pipeline on a **representative subset** (lighting, distance, motion) and adjust thresholds until misses vs. false positives are acceptable.  
3. **Document** the threshold settings (and software version) used for each release batch for auditability.  
4. Perform **human QA** on a random sample of outputs; escalate full manual review for especially sensitive content.  
5. Distribute **only** the redacted exports through channels intended for the public; keep lineage clear so unredacted masters are not accidentally attached to the same bundle.

---

### 5. Limitations (important for legal/compliance language)

- **No detector is 100% recall**; small, distant, or unusual faces/plates may score below threshold or not be detected.  
- **False positives** can blur non-PII (signs, patterns, clothing graphics). Lower thresholds increase recall but also false positives.  
- **Blur is irreversible in the output file**, but if the same scene exists unredacted elsewhere, duplication risk remains a **governance** issue, not solved by this tool alone.  
- **Other PII** (names on jerseys, unique tattoos, street numbers, voice in audio) is **out of scope** unless addressed by additional processes.

---

### 6. Summary

We remove PII from shareable **images and videos** by running **EgoBlur Gen2 TorchScript detectors** (face and license plate) on each frame, filtering predictions with **score and NMS** rules, and applying **strong localized blur** inside the resulting regions before writing the public copy. The method is **algorithmic redaction**, not identification or re-identification. For public distribution, combine this tooling with **clear data-handling policies**, **threshold tuning** for your camera content, and **QA sampling** so stakeholders have an accurate expectation of protection strength.

---

*Document version: 1.0 — aligned with EgoBlur Gen2 usage in this repository (`egoblur-gen2`, OpenCV-based blur in `gen2/script/utils.py`). For authoritative model and license terms, see the upstream [EgoBlur / Project Aria](https://www.projectaria.com/tools/egoblur) materials.*

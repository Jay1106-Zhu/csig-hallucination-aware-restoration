# Phase 0.5 — complete experiment artifacts

User-authorized full public delivery, separate from the unchanged Phase0 release.

Includes every Phase05 experiment artifact: real LQ/GT and synthetic sources, lossless regions, global/pairwise/spatial/multiscale features, 324 lightweight verifier checkpoints, 220 controlled candidates, 275 selective outputs and masks, complete metrics/logs/visualizations/reports, plus shared frozen encoder and evaluation model weights.

Eight ZIP packages plus the member-level artifact manifest and SHA256SUMS. Download with:

```powershell
python publication/phase05/download_phase05.py --extract
```

**Scientific decision remains HOLD.** 55 source images = 5 real + 50 synthetic; 110 human-review rows remain pending. No fabricated human labels and no Phase1/restoration training. The public delivery does not claim the human annotation requirement is complete.

All package hashes are checked against GitHub full-asset SHA256. Anonymous access and sample bytes are independently checked. Host-specific paths are sanitized in exported text; original experiment snapshots and historical source hashes are preserved. Use this release's `artifact_manifest.json` for exported member checksums.

Dataset/model rights remain with their respective owners; this release does not grant new licenses. Original Phase0 release files are unchanged.

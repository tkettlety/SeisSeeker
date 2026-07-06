# AGENTS.md

## Project Focus

SeisSeeker is a research-style Python package for seismic array detection and simple location. When updating the repo, prefer practical usability improvements over broad refactors unless the task clearly calls for deeper algorithm work.

## Main Entry Points

- `SeisSeeker/processing/detection.py`
  - `setup_detection`
  - `run_array_proc()`
  - `detect_events()`
  - `create_location_LUTs()`
  - `locate_events()`
- `SeisSeeker/processing/selby.py`
  - Selby-inspired alternative detector backend

## Data Assumptions

- The waveform loader currently expects:
  - `<archive>/<YYYY>/<MM>/<DD>/<YYYYMMDDTHHMMSS>_<STATION>_<CHANNEL>.mseed`
- Station metadata must include:
  - `Latitude,Longitude,Elevation,Name`
- `channels_to_use` is currently expected in one of these forms:
  - `["??Z"]`
  - `["??Z", "??N", "??E"]`
  - `["??Z", "??1", "??2"]`

## Known Gotchas

- The bundled example archive uses `.msd` files, but the current loader looks for `.mseed`.
- Importing `SeisSeeker.processing` pulls in `lookup_table_manager.py`, so `scikit-fmm` is effectively required for normal package imports.
- The Selby backend is intentionally documented as "Selby-inspired", not as a full generalized `FMP` implementation.
- Location code currently mixes units:
  - LUT velocity model expects metres and m/s
  - `receiver_vp` / `receiver_vs` are used in km/s

## Documentation Preference

Keep the README usage-first:

- how to install
- how to structure a dataset
- how to run detection
- how to run location
- what outputs are produced
- what limitations matter in practice

Avoid turning the README into a paper summary unless the user explicitly asks for that.

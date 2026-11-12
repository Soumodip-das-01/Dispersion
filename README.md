# Dispersion Model Project (Python + Streamlit)

Predicts the affected zone after a toxic/flammable gas release.

## Features (matches assignment guidelines)
- **Takes user input**: release amount/rate, release height, wind speed & direction, stability class, terrain, temperature, pressure, threshold.
- **Concentration at any distance and instant of time**: point calculator (x, y, z, t) and centre-line profile.
- **Contour plot** of concentration with the threshold contour and affected-area statistics.
- **Models**: Gaussian Plume (continuous) and Gaussian Puff (instantaneous), Briggs coefficients (rural/urban).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `dispersion.py`, `requirements.txt` to a GitHub repo.
2. Go to https://share.streamlit.io, click **New app**, choose the repo, set main file = `app.py`, click **Deploy**.

## Files
| File | Purpose |
|------|---------|
| `dispersion.py` | Model equations (Briggs sigmas, plume, puff) |
| `app.py` | Streamlit user interface |
| `requirements.txt` | Dependencies |

# 🌍 CO₂ Land-Cover Estimator

> Estimate whether a random spot on Earth is a carbon **sink** or **source** using satellite imagery, computer vision, and a Net Ecosystem Exchange (NEE) differential equation model.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ What it does

1. **Picks a random land coordinate** on Earth (or uses one you provide).
2. **Fetches a satellite image** via the Google Maps Static API.
3. **Segments the image** into four land-cover types using either:
   - 🎨 **HSV colour-range masks** (fast, offline, default)
   - 🤖 **SegFormer-B2** semantic segmentation (more accurate, needs `torch`)
4. **Solves an ODE** (dC/dt = F_total / h) to estimate the net CO₂ flux over the zone.
5. **Renders a 4-panel figure** showing the satellite image, segmentation overlay, land-cover composition donut chart, and the CO₂ concentration trajectory.

---

## 🧮 The physics — Net Ecosystem Exchange

The model is grounded in the **NEE** framework used in terrestrial ecology:

```
NEE = RE − GPP
```

where **RE** is ecosystem respiration (emits CO₂) and **GPP** is gross primary production (absorbs CO₂ through photosynthesis).

We approximate this as a **linear weighted sum** over land-cover classes:

```
F_total [gC·m⁻²·yr⁻¹] = Σ fᵢ · Fᵢ
```

| Cover type | Fᵢ (gC·m⁻²·yr⁻¹) | Role |
|---|---|---|
| 🌿 Dense vegetation | −400 | Strong **sink** |
| 💧 Water bodies | −20 | Weak sink |
| 🏜️ Arid / bare soil | +20 | Near-neutral |
| 🏙️ Urban | +1500 | Strong **source** |

The ODE governing the boundary-layer CO₂ concentration C (ppm):

```
dC/dt = F_total / (ρ_air · h · MW_ratio)
```

- `ρ_air` = 1.225 kg/m³ (dry air at STP)
- `h` = 1000 m (planetary boundary layer height)
- `MW_ratio` converts gC → ppm_CO₂

The ODE is solved with `scipy.integrate.solve_ivp` (RK45).

> **Disclaimer:** This is a first-order approximation. It does not account for horizontal advection, seasonal variation, or sub-pixel heterogeneity. It is intended for educational exploration.

---

## 🚀 Quick start

### 1. Clone & install

```bash
git clone https://github.com/yourname/co2-landcover-estimator.git
cd co2-landcover-estimator
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and paste your Google Maps Static API key
```

Or export it directly:

```bash
export GOOGLE_MAPS_API_KEY="your_key_here"
```

[Get a free API key →](https://developers.google.com/maps/documentation/maps-static/get-api-key)

### 3. Run

```bash
# Random location (default HSV segmentation)
python main.py

# Fixed location — Cordoba, Argentina
python main.py --lat -31.4 --lon -64.2

# Use SegFormer ML segmentation (needs torch + transformers)
python main.py --mode ml

# Save the output plot
python main.py --output result.png

# All options
python main.py --lat -31.4 --lon -64.2 --mode hsv --zoom 12 --size 512 --years 1 --output out.png
```

---

## 📁 Project structure

```
co2-landcover-estimator/
├── main.py              # Entry point
├── config.py            # All tuneable constants (flux values, HSV thresholds…)
├── requirements.txt
├── .env.example
└── src/
    ├── geo.py           # Random land-coordinate generator
    ├── imagery.py       # Google Maps Static API client
    ├── segmentation.py  # HSV + SegFormer land-cover classifier
    ├── co2_model.py     # NEE ODE model
    └── visualizer.py    # 4-panel matplotlib figure
```

---

## ⚙️ Configuration

All physics constants and thresholds live in **`config.py`**. Edit them to experiment:

```python
# Flux values (gC·m⁻²·yr⁻¹) — change these to test sensitivity
CO2_FLUX_BY_COVER = {
    "vegetation": -400.0,
    "water":       -20.0,
    "arid":        +20.0,
    "urban":      +1500.0,
}

# Boundary-layer height (m) — affects ΔC_ppm magnitude
BOUNDARY_LAYER_HEIGHT_M = 1000.0
```

---

## 🔬 Segmentation modes

### HSV (default `--mode hsv`)
Classifies each pixel by its hue/saturation/value range. Fast and dependency-light. Works best at zoom levels 10–13 where colour signatures are distinct.

### SegFormer (`--mode ml`)
Uses [nvidia/segformer-b2-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) from HuggingFace. The 150 ADE20k semantic labels are mapped to our four land-cover categories. More robust to lighting variation and mixed pixels.

Extra install:
```bash
pip install torch transformers
```

---

## 📊 Example output

```
🌍 Picking a random land coordinate …
📍 Selected: (3.7241, 18.6583)
🛰  Fetching satellite image …
🔍 Segmenting image (mode=hsv) …
   vegetation=61.3%  water=4.2%  arid=18.5%  urban=7.1%
🧮 Solving CO₂ flux ODE …
   Net flux : -242.8 gC/m²/yr  →  SINK ✅
   ΔC (1 yr): -0.0023 ppm over zone
📊 Rendering results …
```

---

## 📚 Scientific references

- Chapin, F.S. III et al. (2011). *Principles of Terrestrial Ecosystem Ecology*. Springer.
- Sitch, S. et al. (2015). Recent trends and drivers of regional sources and sinks of carbon dioxide. *Biogeosciences*, 12, 653–679.
- Churkina, G. et al. (2010). Carbon stored in human settlements: the conterminous United States. *Global Change Biology*, 16, 2296–2309.
- Xiao, J. et al. (2014). Carbon fluxes, evapotranspiration, and water use efficiency of terrestrial ecosystems in China. *Agricultural and Forest Meteorology*, 182–183, 76–90.

---

## 📄 License

MIT — do whatever you want, just keep the attribution.

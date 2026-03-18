# 🌍 CO₂ Land-Cover Estimator

> Analyse any spot on Earth from satellite imagery and estimate whether that zone is a carbon **sink** or **source** — plus how many trees you'd need to plant to offset its emissions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![No API key](https://img.shields.io/badge/Satellite%20imagery-free%2C%20no%20key-brightgreen)

---

## ✨ What it does

1. **Picks a random land coordinate** on Earth, or uses one you provide.
2. **Fetches a satellite image** via ESRI World Imagery tiles — **completely free, no API key, no account needed**.
3. **Segments the image** into four land-cover types (vegetation, water, arid, urban) using a **hybrid model**:
   - HSV colour masks for water and vegetation (very accurate by colour signature)
   - SegFormer-B2 (NVIDIA, HuggingFace) for urban vs. arid distinction
4. **Calibrates the urban CO₂ flux** by querying population density from WorldPop (free REST API), so a suburb in Córdoba is not treated the same as central Tokyo.
5. **Estimates net CO₂ flux** using a Net Ecosystem Exchange (NEE) weighted model.
6. **Calculates how many trees** would need to be planted to offset the zone's emissions (or the tree-equivalent for sinks).
7. **Renders a clear, infographic-style figure** designed to be understood by anyone — from a curious neighbour to a climate scientist.

---

## 🧮 The physics — Net Ecosystem Exchange

The model uses the standard **NEE** framework from terrestrial ecology:

```
F_total [gC·m⁻²·yr⁻¹] = Σᵢ fᵢ · Fᵢ
```

where `fᵢ` is the fractional cover of each land-cover class and `Fᵢ` is its literature flux:

| Cover type | Fᵢ (gC·m⁻²·yr⁻¹) | Role |
|---|---|---|
| 🌿 Dense vegetation | −400 | Strong **sink** |
| 💧 Water bodies | −20 | Weak sink |
| 🏜️ Arid / bare soil | +20 | Near-neutral |
| 🏙️ Urban | **dynamic** | Calibrated by local population density |

The urban flux is not hardcoded — it scales with local population density fetched from WorldPop:

| Density (hab/km²) | Example | Urban flux |
|---|---|---|
| < 500 | Small town / suburb | +200 gC/m²/yr |
| 500–5 000 | Mid-size city (Córdoba) | +600 gC/m²/yr |
| 5 000–20 000 | Dense city (Buenos Aires) | +1 200 gC/m²/yr |
| > 20 000 | Megacity (Tokyo) | +1 500 gC/m²/yr |

### Tree offset

```
trees_needed = total_emission_kgCO₂_yr / 21 kgCO₂_per_tree_yr
forest_ha    = trees_needed / 400 trees_per_ha
```

> **Disclaimer:** First-order approximation. Does not account for horizontal advection, seasonal variation, or sub-pixel heterogeneity. Intended for educational exploration.

---

## 🚀 Quick start

### 1. Clone & install

```bash
git clone https://github.com/yourname/co2-landcover-estimator.git
cd co2-landcover-estimator
pip install -r requirements.txt
```

No API key or account is needed for basic usage.

### 2. Run

```bash
# Random location, neighbourhood scale (default)
python main.py

# Fixed coordinates — centre of Córdoba, Argentina
python main.py --lat -31.393 --lon -64.195

# City-wide view
python main.py --lat -31.393 --lon -64.195 --scale ciudad

# Neighbourhood view with ML segmentation (more accurate, needs torch)
python main.py --lat -31.393 --lon -64.195 --scale barrio --mode ml

# Save the result figure
python main.py --lat -31.393 --lon -64.195 --output result.png
```

### 3. (Optional) Install ML segmentation

For the more accurate hybrid segmentation mode (default when torch is available):

```bash
pip install torch transformers
```

The SegFormer-B2 model (~110 MB) downloads automatically on first use from HuggingFace.

---

## 📁 Project structure

```
co2-landcover-estimator/
├── main.py              # Entry point
├── config.py            # All tuneable constants
├── requirements.txt
├── .env.example
└── src/
    ├── geo.py           # Random land-coordinate generator
    ├── imagery.py       # ESRI World Imagery tile fetcher (no key needed)
    ├── segmentation.py  # HSV + SegFormer hybrid land-cover classifier
    ├── density.py       # WorldPop population density → urban flux calibration
    ├── co2_model.py     # NEE flux model + tree offset estimator
    └── visualizer.py    # Infographic-style result figure
```

---

## ⚙️ Configuration

All physics constants and thresholds live in **`config.py`**:

```python
# Flux values (gC·m⁻²·yr⁻¹)
CO2_FLUX_BY_COVER = {
    "vegetation": -400.0,
    "water":       -20.0,
    "arid":        +20.0,
    "urban":      +600.0,   # overridden at runtime by density.py
}

# Tree sequestration (Nowak et al. 2013)
TREE_CO2_KG_PER_YEAR = 21.0

# Spatial scale presets
SCALE_PRESETS = {
    "barrio":  {"zoom": 15, "label": "Neighbourhood (~500 m)"},
    "ciudad":  {"zoom": 12, "label": "City (~10 km)"},
    "region":  {"zoom": 10, "label": "Region (~40 km)"},
}
```

---

## 🔬 Segmentation modes

| Mode | Description | Requires |
|---|---|---|
| `hybrid` | HSV for water/vegetation + SegFormer for urban/arid. **Best accuracy** (default). | `torch`, `transformers` |
| `hsv` | Colour-range masks only. Fast, offline. | nothing extra |
| `ml` | Full SegFormer segmentation. | `torch`, `transformers` |

---

## 📊 Output

The result figure contains:

- **Satellite image** of the analysed zone
- **Segmented image** with colour-coded land cover (green=vegetation, red=urban, blue=water, yellow=arid)
- **Cover composition** pie chart
- **Zone summary** cards: CO₂/yr, tree equivalence, forest area needed
- **Greenhouse impact thermometer** — vertical scale from dense forest (green) to megacity (red)
- **Area comparison** — proportional squares showing the analysed zone vs the forest needed to compensate

---

## 📚 Scientific references

- Chapin, F.S. III et al. (2011). *Principles of Terrestrial Ecosystem Ecology*. Springer.
- Sitch, S. et al. (2015). Biogeosciences, 12, 653–679.
- Churkina, G. et al. (2010). Global Change Biology, 16, 2296–2309.
- Kennedy, C. et al. (2011). Journal of Industrial Ecology, 15(1), 68–83.
- Nowak, D.J. et al. (2013). Urban Forestry & Urban Greening, 12(4), 490–495.
- WorldPop (2020). Global High Resolution Population Denominators Project. doi:10.5258/SOTON/WP00647

---

## 📄 License

MIT — do whatever you want, just keep the attribution.
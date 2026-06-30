import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['STPSF_PATH'] = '/Users/epadilla/Library/CloudStorage/OneDrive-Personal/stpsf-data'

import slsim
from slsim.Lenses.lens_pop import LensPop
from slsim.Pipelines.roman_speclite import configure_roman_filters
from slsim.Pipelines.roman_speclite import filter_names
import speclite
from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
import slsim.Sources as sources
import slsim.Deflectors as deflectors
import slsim.Pipelines as pipelines
from slsim.ImageSimulation.roman_image_simulation import simulate_roman_image
from slsim.Lenses.lens import Lens
import random
import pickle
import gc
import psutil
from pathlib import Path
from datetime import datetime
from astropy.table import vstack

import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r"invalid value encountered in do_format",
)
from scipy.integrate import IntegrationWarning

# suppress all the warnings you're seeing
warnings.filterwarnings("ignore", category=IntegrationWarning)
warnings.filterwarnings("ignore", message=r".*JWST PRD VERSION.*")
warnings.filterwarnings("ignore", message=r".*pysiaf.*")
warnings.filterwarnings("ignore", message=r".*Angular size is converted.*")
warnings.filterwarnings("ignore", message=r".*invalid value encountered.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r".*attach.dylib.*")

# ── helpers ───────────────────────────────────────────────────────────────────
def print_memory(label=""):
    mem = psutil.virtual_memory()
    used  = (mem.total - mem.available) / 1e9
    total = mem.total / 1e9
    print(f"[{label}] RAM: {used:.1f} / {total:.1f} GB ({mem.percent:.0f}%)")


def deduplicate_lenses(all_lenses, tol=1e-6):
    """Remove duplicate lenses based on key physical properties."""
    seen = set()
    unique_lenses = []
    duplicates = 0

    for lens in all_lenses:
        # build a fingerprint from key properties
        key = (
            round(lens.deflector_redshift, 4),
            round(lens.source_redshift_list[0], 4),
            round(float(lens.einstein_radius[0]), 4),
        )

        if key not in seen:
            seen.add(key)
            unique_lenses.append(lens)
        else:
            duplicates += 1

    print(f"Total before dedup: {len(all_lenses)}")
    print(f"Duplicates removed: {duplicates}")
    print(f"Unique lenses:      {len(unique_lenses)}")
    return unique_lenses

# ── config ────────────────────────────────────────────────────────────────────
survey = 'time_domain_deep'

if survey == 'time_domain_wide':
    filters = ["F062", "F087", "F106", "F129", "F158"]
    kwargs_variability = {"supernovae_lightcurve", "F062", "F087", "F106", "F129", "F158"}
elif survey == "time_domain_deep":
    filters = ["F087", "F106", "F129", "F158", "F184"]
    kwargs_variability = {"supernovae_lightcurve", "F087", "F106", "F129", "F158", "F184"}

seed = 42
np.random.seed(seed)
random.seed(seed)

path = os.path.dirname(slsim.__file__)
module_path, _ = os.path.split(path)
skypy_config = os.path.join(module_path, "data/SkyPy/roman-like.yml")

configure_roman_filters()
roman_filters = filter_names()
speclite.filters.load_filters(*roman_filters)   # unpacks all 8 at once

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

kwargs_deflector_cut = {"band": "F129", "band_max": 25, "z_min": 0.01, "z_max": 3.0}
kwargs_source_cut    = {"z_min": 0.01, "z_max": 5}
kwargs_lens_cut      = {"min_image_separation": 0.5, "max_image_separation": 10}
time_range           = np.linspace(-50, 500, 550)

kwargs_sn = {
    "variability_model": "light_curve",
    "kwargs_variability": kwargs_variability,
    "sn_type": "Ia",
    "sn_absolute_mag_band": "bessellb",
    "sn_absolute_zpsys": "ab",
    "lightcurve_time": np.linspace(-50, 100, 31),
    "sn_modeldir": None,
}

chunk_dir = Path("/Users/epadilla/Library/CloudStorage/OneDrive-Personal/slsim/notebooks/chunks")
chunk_dir.mkdir(parents=True, exist_ok=True)

# ── chunked generation ────────────────────────────────────────────────────────
target_lenses  = 10000
chunk_sky_area = 1000   # deg2 per chunk
all_lenses     = []
#chunk_idx      = 0

existing_chunks = sorted(chunk_dir.glob("chunk_*.pkl"))

if existing_chunks:
    print(f"Found {len(existing_chunks)} existing chunks, loading...")
    for p in existing_chunks:
        with open(p, "rb") as f:
            all_lenses.extend(pickle.load(f))
    chunk_idx = len(existing_chunks)  # start from where we left off
    print(f"Recovered {len(all_lenses)} lenses from {len(existing_chunks)} chunks")
else:
    chunk_idx = 0
    print("No existing chunks found, starting fresh")

print(f"Starting from chunk {chunk_idx+1}, {len(all_lenses)} lenses so far")



while len(all_lenses) < target_lenses:
    print(f"\n--- Chunk {chunk_idx+1} | Total so far: {len(all_lenses)} / {target_lenses} ---")
    print_memory(f"chunk {chunk_idx+1} start")

    deflector_sky_area = Quantity(value=5,   unit="deg2")
    source_sky_area    = Quantity(value=10,  unit="deg2")
    host_sky_area      = Quantity(value=2,   unit="deg2")
    sky_area           = Quantity(value=chunk_sky_area, unit="deg2")

    print("  Building galaxy pipeline...")

    galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
        skypy_config=skypy_config, sky_area=deflector_sky_area, filters=None, cosmo=cosmo
    )
    print("  Done. Building host pipeline...")

    galaxy_simulation_pipeline_host = pipelines.SkyPyPipeline(
        skypy_config=skypy_config, sky_area=host_sky_area, filters=None, cosmo=cosmo
    )
    print("  Done. Building deflectors...")

    lens_galaxies = deflectors.EllipticalLensGalaxies(
        galaxy_list=galaxy_simulation_pipeline.red_galaxies,
        kwargs_cut=kwargs_deflector_cut,
        kwargs_mass2light={},
        cosmo=cosmo,
        sky_area=deflector_sky_area,
    )
    print("  Done. Building supernovae catalog...")

    supernovae_catalog = sources.SupernovaeCatalog.SupernovaeCatalog(
        sn_type="Ia",
        band_list=filters,
        lightcurve_time=time_range,
        absolute_mag_band="bessellb",
        absolute_mag=None,
        mag_zpsys="ab",
        cosmo=cosmo,
        skypy_config=skypy_config,
        sky_area=source_sky_area,
        host_galaxy_candidate=galaxy_simulation_pipeline_host.blue_galaxies[0:2000],
    )
    supernovae_data = supernovae_catalog.supernovae_catalog(
        host_galaxy=True, lightcurve=False
    )
    print("  Done. Building source population...")
    source_SNIa = sources.PointPlusExtendedSources(
        point_plus_extended_sources_list=supernovae_data,
        cosmo=cosmo,
        sky_area=source_sky_area,
        kwargs_cut={},
        catalog_type="skypy",
        source_size=None,
        extended_source_type="single_sersic",
        point_source_type="supernova",
        point_source_kwargs=kwargs_sn,
    )

    lens_pop = LensPop(
        deflector_population=lens_galaxies,
        source_population=source_SNIa,
        cosmo=cosmo,
        sky_area=sky_area,
    )
    print("  Done. Drawing lens population...") 
    chunk = lens_pop.draw_population(             # consistent name: chunk
        kwargs_lens_cuts=kwargs_lens_cut, speed_factor=10
    )

    print(f"Chunk {chunk_idx+1} yielded {len(chunk)} lenses")
    all_lenses.extend(chunk)                      # append to running list

    # save chunk as safety net
    chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.pkl"
    with open(chunk_path, "wb") as f:
        pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved chunk to {chunk_path}")

    # free memory
    del galaxy_simulation_pipeline, galaxy_simulation_pipeline_host
    del lens_galaxies, supernovae_catalog, supernovae_data
    del source_SNIa, lens_pop, chunk
    gc.collect()
    print_memory(f"chunk {chunk_idx+1} after cleanup")

    chunk_idx += 1

print(f"\nDone! Total lenses collected: {len(all_lenses)}")
all_lenses = deduplicate_lenses(all_lenses)   # deduplicate once here
print(f"Total lenses after dedup: {len(all_lenses)}")


# ── final save ────────────────────────────────────────────────────────────────
out = {
    "lens_population": all_lenses,              # save all_lenses not lens_population
    "meta": {
        "created_utc": datetime.utcnow().isoformat(),
        "n_lenses": len(all_lenses),
        "n_chunks": chunk_idx,
        "chunk_sky_area_deg2": chunk_sky_area,
        "slsim_version": getattr(slsim, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "seed_numpy": seed,
        "seed_python": seed,
        "notes": "roman_supernovae_population_deep 10k sample via chunking",
    },
}

save_path = Path("/Users/epadilla/Library/CloudStorage/OneDrive-Personal/slsim/roman_deep_lens_population_10k.pkl")
save_path.parent.mkdir(parents=True, exist_ok=True)

with open(save_path, "wb") as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL) #remember to save the pickle file in a flexible numpy

print(f"Saved {len(all_lenses)} lenses to {save_path}")
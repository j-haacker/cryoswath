"""CryoSat L1b to L2 processing tutorial"""

from cryoswath.l1b import (
    append_best_fit_phase_index,
    append_ambiguous_reference_elevation,
    read_esa_l1b,
    to_l2,
)
from cryoswath.misc import download_dem
from cryoswath.test_plots.waveform import dem_transect
import geopandas as gpd
import matplotlib.pyplot as plt
import mpld3
from pathlib import Path
import requests
from shapely import box
import streamlit as st
import streamlit.components.v1 as components


@st.cache_resource()
def _process(coh, pwr, poca_upper, swath_start_window, smooth):
    if not Path("CS_LTA__SIR_SIN_1B_20191206T003639_20191206T003855_E001.nc").exists():
        response = requests.get("https://science-pds.cryosat.esa.int/?do=download&file=Cry0Sat2_data%2FSIR_SIN_L1%2F2019%2F12%2FCS_LTA__SIR_SIN_1B_20191206T003639_20191206T003855_E001.nc")
        if response.status_code != 200:
            raise Exception(f"Failed to get track. Code {response.status_code}.")
        with open("CS_LTA__SIR_SIN_1B_20191206T003639_20191206T003855_E001.nc", "wb") as f:
            f.write(response.content)
    if not Path("arcticdem-mosaics-v4.1-32m.zarr").exists():
        download_dem(gpd.GeoSeries(box(586700.0, -2197200.0, 694900.0, -2138600.0), crs=3413))
    ds = (
        read_esa_l1b(
            "CS_LTA__SIR_SIN_1B_20191206T003639_20191206T003855_E001.nc",
            drop_outside=False,
            waveform_selection=585,
            coherence_threshold=coh,
            power_threshold=("snr", pwr),
            smooth_phase_difference=smooth,
            swath_start_kwargs={"poca_upper": poca_upper, "swath_start_window": swath_start_window}
        )
        .pipe(append_ambiguous_reference_elevation, "arcticdem-mosaics-v4.1-32m.zarr")
        .pipe(append_best_fit_phase_index)
    )
    result = to_l2(ds, retain_vars={"xph_dists": "off_nadir", "ph_diff_waveform_20_ku": "ph_diff"})
    return ds, result


st.markdown("""
<style>
iframe {
    background-color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("CryoSat L1b to L2 processing tutorial")
st.write("""This tutorial aims to show the impacts of processing choices. More
advanced choices, e.g., concerning how the phase difference ambiguity is
solved, cannot be made here. Please file GitHub issues to request
features.
""")
with st.form(key="input"):
    st.slider("Coherence threshold", 0.0, 1.0, 0.6, 0.01, key="coh")
    st.slider("Power threshold (signal to noise ratio)", 0, 100, 10, 1, key="pwr")
    left, mid, right = st.columns(3)
    left.slider("POCA latest after suff. coh. (m)", 0, 50, 10, 1, key="poca_upper")
    mid.slider("Swath start earliest after POCA (m)", 0, 15, 5, 1, key="start_lower")
    right.slider("Swath start range (m)", -1, 100, 50, 1, help="Set -1 to process entire waveform", key="start_range")
    left, right = st.columns(2)
    left.slider("Smoothing window extent", 0, 50, 21, 1, key="window")
    right.slider("Smoothing standard deviation", 0.0, 10.0, 5.0, 0.1, key="std")
    st.form_submit_button("Process")
# print(list(ds.data_vars.keys()))
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, ax = plt.subplots(figsize=(720*px, 540*px))
st.toggle("Show all possible solutions", help="Activate and zoom out to see ambiguous solutions", key="show_other")
ds, result = _process(
    st.session_state.coh,
    st.session_state.pwr,
    st.session_state.poca_upper,
    (0, -1) if st.session_state.start_range < 0
    else (st.session_state.start_lower, st.session_state.start_range),
    False
    if st.session_state.window == 0 or st.session_state.std == 0
    else {"window_extent": st.session_state.window, "std": st.session_state.std},
)
dem_transect(ds, ax=ax, selected_phase_only=not st.session_state.show_other)
# result = to_l2(ds, out_vars={"xph_elevs": "height", "xph_dists": "off_nadir"})
plt.xlim(result.off_nadir.min() - 2_000, result.off_nadir.max() + 2_000)
plt.ylim(result.height.min() - 25, result.height.max() + 25)
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600)

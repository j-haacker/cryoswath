"""CryoSat L1b to L2 processing tutorial"""

from cryoswath.l1b import (
    append_best_fit_phase_index,
    append_ambiguous_reference_elevation,
    read_esa_l1b,
    to_l2,
)
from cryoswath.test_plots.waveform import dem_transect
import matplotlib.pyplot as plt
import mpld3
import streamlit as st
import streamlit.components.v1 as components


@st.cache_resource()
def _process(coh, pwr, smooth):
    ds = (
        read_esa_l1b(
            "CS_OFFL_SIR_SIN_1B_20191206T003639_20191206T003855_D001.nc",
            drop_outside=False,
            waveform_selection=585,
            coherence_threshold=coh,
            power_threshold=("snr", pwr),
            smooth_phase_difference=smooth,
        )
        .pipe(append_ambiguous_reference_elevation, "./mini_dem.tif")
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
st.write("This tutorial aims to show the impacts of processing choices.")
with st.form(key="input"):
    st.slider("Coherence threshold", 0.0, 1.0, 0.6, 0.01, key="coh")
    st.slider("Power threshold (signal to noise ratio)", 0, 100, 10, 1, key="pwr")
    st.toggle("Smooth phase difference", False, key="smooth")
    st.form_submit_button("Process")
# print(list(ds.data_vars.keys()))
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, ax = plt.subplots(figsize=(720*px, 540*px))
st.toggle("Show all possible solutions", key="show_other")
ds, result = _process(st.session_state.coh, st.session_state.pwr, st.session_state.smooth)
dem_transect(ds, ax=ax, selected_phase_only=not st.session_state.show_other)
# result = to_l2(ds, out_vars={"xph_elevs": "height", "xph_dists": "off_nadir"})
plt.xlim(result.off_nadir.min() - 300, result.off_nadir.max() + 300)
plt.ylim(result.height.min() - 5, result.height.max() + 5)
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600)

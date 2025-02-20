Tests
=====

In the directory ``tests/reports`` you can find notebooks that are build to evaluate cryoswath.
If you modify the core components of cryoswath, which you are encouraged to do(!), you shoud run the notebooks to verify that your results are resonable.
This test is only a first step.
If you are satisfied, do a broader validation campaign.

`tests/reports/l1b_swath_start.ipynb
<https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l1b_swath_start.ipynb>`_
tests edge cases for finding the start of the swath domain. This is 

`tests/reports/l1b_waveform.ipynb
<https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l1b_waveform.ipynb>`_
shows the estimated surface elevations for a waveform overlayed by the
crosssection of the glacier.

`tests/reports/l2_dem_comparison.ipynb
<https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l2_dem_comparison.ipynb>`_
compares many elevation estimates to a reference elevation model.

`tests/reports/l2_tested_data_comparison.ipynb
<https://github.com/j-haacker/cryoswath/blob/main/tests/reports/l2_tested_data_comparison.ipynb>`_
compares the elevation estimates against the results of cryoswath's
mother implementation that was thoroughly tested.

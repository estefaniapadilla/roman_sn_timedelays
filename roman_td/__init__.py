"""
roman_td
========
Time-delay measurement for gravitationally lensed supernovae in the
Roman High Latitude Time Domain Survey (HLTDS).

Modules
-------
sntd_wrapper   : SALT2-extended + SNTD baseline fitter  (~1-5 min/system)
bayesn_wrapper : BayeSN + SNTD two-stage fitter         (~5-30 min/system, better dust)
simulate       : Multi-image Roman photometry simulation from slsim lens populations
crosscorr      : [TODO] GP cross-correlator for sub-minute population-scale estimation
"""

from roman_td.sntd_wrapper import (
    register_roman_bands,
    measure_one,
    DEFAULT_DEPTH_5SIG,
)

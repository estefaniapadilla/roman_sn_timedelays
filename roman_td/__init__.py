"""
roman_td
========
Time-delay measurement for gravitationally lensed supernovae in the
Roman High Latitude Time Domain Survey (HLTDS).

Modules
-------
sntd_wrapper   : SALT2-extended + SNTD fitter, GP-primed "gp" mode
bayesn_wrapper : BayeSN + SNTD two-stage fitter (better dust handling)
simulate       : multi-image Roman photometry simulation from slsim
crosscorr      : Stage-1 GP cross-correlator (seconds/system)
tokenize       : Stage-2 transformer input preparation (pure numpy)
paths          : repo data/output locations

NOTE: no eager submodule imports here. The fitting stack (sncosmo, sntd,
bayesn) only exists in the `sntd_bayesn` env, while the ML env imports
only the lightweight modules (tokenize, paths) — an eager import of
sntd_wrapper would make `import roman_td.tokenize` fail there. Import
what you need explicitly, e.g.:

    from roman_td.sntd_wrapper import measure_one
    from roman_td.tokenize import tokenize_system
"""

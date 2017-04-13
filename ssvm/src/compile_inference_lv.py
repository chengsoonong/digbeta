import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()}, inplace=True)
import inference_lv  # noqa: F401

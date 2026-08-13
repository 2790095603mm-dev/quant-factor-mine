"""规模族：流通市值（A 股小市值效应历史上显著但已衰减，需谨慎）"""

import numpy as np
import pandas as pd

from qfm.factors.base import register_factor


@register_factor("ln_mv_float", "规模", "流通市值（对数），小市值效应已衰减，常与其他因子正交化使用", "negative")
def ln_mv_float(d) -> pd.DataFrame:
    return np.log(d.mv_float)

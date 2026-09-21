"""Multi-Factor Lab 的持久化模型与注册表。"""

from qfm.multifactor.models import CompositeDefinition, WEIGHT_MODES
from qfm.multifactor.registry import CompositeRegistry

__all__ = ["CompositeDefinition", "CompositeRegistry", "WEIGHT_MODES"]

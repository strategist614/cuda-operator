
from dataclasses import dataclass

@dataclass
class TensorType:
    dtype: str
    shape: list

    def __str__(self):
        return f"tensor<{self.shape}x{self.dtype}>"

@dataclass
class PointerType:
    dtype: str = "ptr"

    def __str__(self):
        return "ptr"

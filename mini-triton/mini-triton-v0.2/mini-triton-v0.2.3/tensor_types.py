from dataclasses import dataclass

@dataclass
class TensorType:
    dtype: str
    shape: tuple

    def __str__(self):
        return f"tensor<{self.shape}x{self.dtype}>"

@dataclass
class PointerType:
    def __str__(self):
        return "ptr"

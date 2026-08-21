
from dataclasses import dataclass

@dataclass
class TensorType:
    shape: tuple
    dtype: str
    memory: str = "global"

    def __str__(self):
        return f"tensor<{self.shape}x{self.dtype}, {self.memory}>"


@dataclass
class RegisterType:
    dtype: str

    def __str__(self):
        return self.dtype

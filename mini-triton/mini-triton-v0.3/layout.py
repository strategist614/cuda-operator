
from dataclasses import dataclass

@dataclass
class BlockedLayout:
    threads: int
    elements_per_thread: int

    def __str__(self):
        return (
            f"blocked("
            f"threads={self.threads},"
            f"elements={self.elements_per_thread})"
        )


@dataclass
class LayoutTensorType:
    dtype: str
    shape: tuple
    layout: object

    def __str__(self):
        return (
            f"tensor<{self.shape}x{self.dtype},"
            f" {self.layout}>"
        )

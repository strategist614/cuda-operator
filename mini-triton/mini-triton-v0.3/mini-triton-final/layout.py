
from dataclasses import dataclass


@dataclass
class BlockedLayout:
    warps: int
    threads_per_warp: int
    elements_per_thread: int

    @property
    def threads(self):
        return self.warps*self.threads_per_warp

    def __str__(self):
        return (
            f"blocked("
            f"warps={self.warps},"
            f"threads={self.threads},"
            f"elements/thread={self.elements_per_thread})"
        )

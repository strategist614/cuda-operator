from dataclasses import dataclass

@dataclass
class LaunchConfig:
    threads_per_block:int
    warps:int

    def __str__(self):
        return f"block={self.threads_per_block}, warps={self.warps}"


@dataclass
class ThreadMapping:
    threads:int
    elements_per_thread:int

    def element_to_thread(self,e):
        return e // self.elements_per_thread

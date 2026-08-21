from dataclasses import dataclass

@dataclass
class BlockedLayout:
    threads:int
    elements_per_thread:int

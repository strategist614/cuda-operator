
from dataclasses import dataclass

@dataclass
class Register:
    name: str
    dtype: str

    def __str__(self):
        return f"{self.name}:{self.dtype}"


@dataclass
class RegisterMapping:
    elements_per_thread: int

    def map(self, thread_id):
        base = thread_id * self.elements_per_thread
        return [base+i for i in range(self.elements_per_thread)]

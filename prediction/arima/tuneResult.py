from typing import Tuple


class TuneResult:
    def __init__(self, error: float, time: float, order: Tuple):
        self.error = error
        self.time = time
        self.order = order

    def __repr__(self):
        return f"{self.error}; {self.time}s; {self.order}"

    def __str__(self):
        return f"{self.error}; {self.time}s; {self.order}"

    def toString(self):
        return f"{self.error}; {self.time}s; {self.order}"

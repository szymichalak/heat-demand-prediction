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


def stringRowToTuneResult(row: str) -> TuneResult:
    rowArray = row.replace('\n', '').replace(' ', '').replace('(', '').replace(')', '').replace('s', '').split(';')
    return TuneResult(float(rowArray[0]), float(rowArray[1]), tuple([int(o) for o in rowArray[2].split(',')]))

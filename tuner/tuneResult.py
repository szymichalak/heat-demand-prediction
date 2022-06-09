from typing import List


class TuneResult:
    def __init__(self, row: str):
        self.__row = row

        self.error: float = float(self.__row.split(',')[0])
        self.time: float = float(self.__row.split(',')[1])
        self.parameters: List = self.__row.split(',')[2:]
        self.__convertParameters()

    def __convertParameters(self):
        for i, param in enumerate(self.parameters):
            try:
                self.parameters[i] = int(param)
            except ValueError:
                if param == "None":
                    self.parameters[i] = None
                else:
                    self.parameters[i] = param

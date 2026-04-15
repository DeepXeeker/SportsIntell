from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunningMOTStats:
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    gt: int = 0
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0
    deta: float = 0.0
    assa: float = 0.0
    count: int = 0

    def update(self, fp: int, fn: int, idsw: int, gt: int, idtp: int, idfp: int, idfn: int, deta: float, assa: float) -> None:
        self.fp += fp
        self.fn += fn
        self.idsw += idsw
        self.gt += gt
        self.idtp += idtp
        self.idfp += idfp
        self.idfn += idfn
        self.deta += deta
        self.assa += assa
        self.count += 1

    def summary(self) -> dict[str, float]:
        mota = 1.0 - (self.fn + self.fp + self.idsw) / max(self.gt, 1)
        idf1 = (2 * self.idtp) / max(2 * self.idtp + self.idfp + self.idfn, 1)
        deta = self.deta / max(self.count, 1)
        assa = self.assa / max(self.count, 1)
        hota = (deta * assa) ** 0.5
        return {
            "MOTA": mota * 100.0,
            "IDF1": idf1 * 100.0,
            "HOTA": hota * 100.0,
            "DetA": deta * 100.0,
            "AssA": assa * 100.0,
            "IDSW": float(self.idsw),
            "FP": float(self.fp),
            "FN": float(self.fn),
        }

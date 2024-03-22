from dataclasses import dataclass,field

@dataclass
class Cluster:
    id: int
    mean: float
    gravity: float
    std: float
    cov_matrix: float
    members: list
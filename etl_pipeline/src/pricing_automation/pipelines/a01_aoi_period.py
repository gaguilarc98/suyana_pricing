from .utils import *
from dataclasses import dataclass, field, fields, asdict
from typing import Optional
import datetime

#——————————————————————————————————————————
# AOI DEFINITION
#——————————————————————————————————————————

@dataclass
class BoundingBox:
    maxy: float
    miny: float
    minx: float
    maxx: float

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in fields(cls)}
        missing = valid_keys - set(d.keys())
        if missing:
            raise ValueError(f"AOI dict is missing keys: {missing}")
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_era5_area(self) -> list:
        return [self.maxy, self.minx, self.miny, self.maxx]

    def to_dict(self) -> dict:
        return {"maxy": self.maxy, "miny": self.miny, "minx": self.minx, "maxx": self.maxx}

    def __repr__(self) -> str:
        return f"BoundingBox(N={self.maxy}, S={self.miny}, W={self.minx}, E={self.maxx})"
    

def round_up_values(value, part_per_unit):
    return float(np.ceil(value*part_per_unit)/part_per_unit)

def round_down_values(value, part_per_unit):
    return float(np.floor(value*part_per_unit)/part_per_unit)
    
def get_bounds(minx, maxx, maxy, miny, nodes_by_deg=10, **kwargs):
    '''Get bounds that are compatible with the patch size of model'''
    # Return the bounds
    dict_params = {
        'minx': round_down_values(minx, nodes_by_deg),
        'miny': round_down_values(miny, nodes_by_deg),
        'maxx': round_up_values(maxx, nodes_by_deg), # Substract one node since the middle point is included
        'maxy': round_up_values(maxy, nodes_by_deg), # Substract one node since the middle point is included
    }
    return dict_params



#——————————————————————————————————————————
# PERIOD DEFINITION
#——————————————————————————————————————————

def compute_period(reference_year: Optional[int] = None, window: int = 30) -> tuple:
    if reference_year is None:
        reference_year = datetime.date.today().year-1
    end_year = int(round_down_values(reference_year, 1))
    start_year = end_year - window
    return (start_year, end_year)


def get_year_list(start_year: int, end_year: int) -> list:
    return list(range(start_year, end_year + 1))


def describe_period(start_year: int, end_year: int) -> str:
    return f"{start_year}-{end_year} ({end_year - start_year}-year period)"
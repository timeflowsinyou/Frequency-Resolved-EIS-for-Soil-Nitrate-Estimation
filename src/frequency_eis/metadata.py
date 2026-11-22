"""Frozen public specimen metadata and confirmed metadata corrections."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Specimen:
    specimen_id: str
    filename: str
    cohort: str
    density: float
    water_content: float
    concentration_factor: float

ORIGINAL_SPECIMENS = (
    Specimen("M5_01", "M05%_01.mat", "original", 1.25, 0.055, 1.187),
    Specimen("M10_01", "M10%_01.mat", "original", 1.25, 0.1136, 1.14),
    Specimen("M10_02", "M10%_02.mat", "original", 1.31, 0.1187, 1.187),
    Specimen("M10_03", "M10%_03.mat", "original", 1.52, 0.1382, 1.4),
    Specimen("M10_04", "M10%_04.mat", "original", 1.65, 0.15, 1.5),
    Specimen("M15_02", "M15%_02.mat", "original", 1.35, 0.1761, 1.17),
    Specimen("M15_03", "M15%_03.mat", "original", 1.24, 0.1623, 1.08),
    Specimen("V5_01", "V05%_01.mat", "original", 1.1, 0.05, 1.05),
    Specimen("V5_02", "V05%_02.mat", "original", 1.2, 0.05, 1.15),
    Specimen("V5_04", "V05%_04.mat", "original", 1.4, 0.05, 1.15),
    Specimen("V5_05", "V05%_05.mat", "original", 1.5, 0.05, 1.25),
    Specimen("V5_06", "V05%_06.mat", "original", 1.35, 0.05, 1.1),
    Specimen("V10_01", "V10%_01.mat", "original", 1.1, 0.1, 1.0),
    Specimen("V10_03", "V10%_03.mat", "original", 1.3, 0.1, 1.2),
    Specimen("V10_04", "V10%_04.mat", "original", 1.4, 0.1, 1.3),
    Specimen("V10_05", "V10%_05.mat", "original", 1.5, 0.1, 1.4),
    Specimen("V15_01", "V15%_01.mat", "original", 1.2, 0.15, 1.05),
    Specimen("V15_02", "V15%_02.mat", "original", 1.3, 0.15, 1.15),
    Specimen("V15_03", "V15%_03.mat", "original", 1.4, 0.15, 1.25),
    Specimen("V15_05", "V15%_05.mat", "original", 1.45, 0.15, 1.1),
    Specimen("V18.5_01", "V18%_01.mat", "original", 1.35, 0.197, 1.07),
)

HISTORICAL_TRANSFER_SPECIMENS = (
    Specimen("Z05_03", "Z05_03.mat", "historical_transfer", 1.3, 0.05, 1.25),
    Specimen("Z10_02", "Z10_02.mat", "historical_transfer", 1.2, 0.10, 1.10),
    Specimen("Z15_04", "Z15_04.mat", "historical_transfer", 1.5, 0.15, 1.35),
)

ALL_SPECIMENS = ORIGINAL_SPECIMENS + HISTORICAL_TRANSFER_SPECIMENS
ORIGINAL_STAGE_MULTIPLIERS = np.array([0, 2, 4, 6, 8, 10], dtype=np.float64)
TRANSFER_STAGE_MULTIPLIERS = np.array([0.01, 2, 4, 6, 8, 10], dtype=np.float64)
N_STAGES = 6
N_REPEATS = 20
N_FREQUENCIES = 201

# One-based specimen number, one-based stage, field, corrected value.
METADATA_CORRECTIONS = (
    (4,3,"density",1.65),(5,3,"density",1.52),(6,2,"water_content",0.164),
    (6,3,"water_content",0.170),(6,4,"water_content",0.150),(6,5,"water_content",0.156),
    (7,1,"water_content",0.167),(7,2,"water_content",0.174),(7,3,"water_content",0.167),
    (7,6,"water_content",0.170),(11,2,"water_content",0.05),(11,6,"water_content",0.094),
    (12,1,"water_content",0.106),(12,4,"water_content",0.084),(12,5,"water_content",0.095),
    (12,6,"water_content",0.112),(20,3,"water_content",0.179),(20,4,"water_content",0.206),
    (20,5,"water_content",0.200),(20,6,"water_content",0.187),(16,6,"water_content",0.160),
    (17,1,"water_content",0.158),(17,4,"water_content",0.162),(17,6,"water_content",0.171),
    (18,5,"water_content",0.134),(18,6,"water_content",0.152),(19,1,"water_content",0.158),
    (19,2,"water_content",0.157),(19,5,"water_content",0.159),(19,6,"water_content",0.144),
    (9,1,"water_content",0.045),
)

def corrected_conditions(specimen_index: int, stage_index: int) -> tuple[float,float]:
    specimen=ORIGINAL_SPECIMENS[specimen_index]; density=specimen.density; water=specimen.water_content
    for subject,stage,field,value in METADATA_CORRECTIONS:
        if subject==specimen_index+1 and stage==stage_index+1:
            if field=="density": density=value
            else: water=value
    return density,water


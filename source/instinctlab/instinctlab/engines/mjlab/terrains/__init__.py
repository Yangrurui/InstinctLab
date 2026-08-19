"""InstinctMJ's terrain stack, vendored for the mjlab adapter.

This package imports mjlab. Adapter modules must reach it only from builder bodies, the same
rule as every other engine import in ``engines/mjlab``. Virtual obstacles and mesh/STL tiles
are not here: locomotion rough only needs the Perlin height-field grid and the importer that
honors ``class_type``.
"""

from .terrain_generator_cfg import FiledTerrainGeneratorCfg
from .terrain_importer import TerrainImporter
from .terrain_importer_cfg import TerrainImporterCfg

__all__ = ["FiledTerrainGeneratorCfg", "TerrainImporter", "TerrainImporterCfg"]

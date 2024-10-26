from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from schemas.location import Location


class LocationConfigHandler:

    def __init__(self, config_file: Path):
        if not config_file.exists():
            logger.error(f"Config file {config_file} does not exist.")
            raise FileNotFoundError(f"Config file {config_file} does not exist.")

        if config_file.suffix != ".yaml":
            logger.error(f"Config file {config_file} is not a yaml file.")
            raise ValueError(f"Config file {config_file} is not a yaml file.")

        self.config_file = config_file

    def parse_locations(self) -> list[Location]:
        """
        Parse the locations from the config file, e.g. for the capitols.yaml file.
        """

        logger.info(f"Reading locations from {self.config_file}")
        with Path(self.config_file).open("r") as f:
            capitols_config = yaml.safe_load(f)

        try:
            locations = [
                Location(name=name, x=coords["x_coord"], y=coords["y_coord"])
                for name, coords in capitols_config["capitols"].items()
            ]
        except KeyError as e:
            logger.error(f"Could not parse locations from {self.config_file}: {e}")
            raise ValidationError(
                f"Could not parse locations from {self.config_file}: {e}"
            ) from e

        return locations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass
class PathsConfig:
    raw_data_dir: Path | str
    output_dir: Path | str
    image_dir: Path | str

    def __post_init__(self):
        self.raw_data_dir = Path(self.raw_data_dir)
        self.output_dir = Path(self.output_dir)
        self.image_dir = Path(self.image_dir)

@dataclass
class Config:
    paths: PathsConfig

def load_config(config_path: Path) -> Config:
    """Load configuration from TOML file."""
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    return Config(
        paths=PathsConfig(**data["paths"]),
    )

# Usage
if __name__ == "__main__":
    config = load_config(Path("config.toml"))
    print(f"Paths: {config.paths}")

from dataclasses import dataclass
from pathlib import Path

import tomllib
import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

@dataclass
class PathsConfig:
    """Paths to data, output and images directories."""

    raw_data_dir: Path
    output_dir: Path
    image_dir: Path

    def __post_init__(self) -> None:
        """Convert paths into pathlib.Path objects."""
        self.raw_data_dir = Path(self.raw_data_dir)
        self.output_dir = Path(self.output_dir)
        self.image_dir = Path(self.image_dir)

@dataclass
class ReprojectionConfig:
    """Settings for reprojection."""

    best_cell_size: int = 0

@dataclass
class Config:
    """A class containing every dataclass."""

    paths: PathsConfig
    reprojection: ReprojectionConfig

_config_cache: Config | None = None

def get_project_root() -> Path:
    """Get the project root directory, where config.toml lives."""
    # From src/pumpkinpy/config.py, go up 2 levels to reach project root
    return Path(__file__).parent.parent.parent

def load_config() -> Config:
    """Load configuration from config.toml."""
    global _config_cache  # noqa: PLW0603

    if _config_cache is not None:
        return _config_cache
    config_path = get_project_root() / "config.toml"

    if not config_path.exists():
        console.print(
            Panel(
                f"[red]Configuration file not found![/red]\n\n"
                f"Expected location: [yellow]{config_path}[/yellow]\n\n"
                f"Please create a config.toml file based on the example.",
                title="Config Error",
                border_style="red"
            )
        )
        raise typer.Exit(code=1)

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    _config_cache = Config(
        paths=PathsConfig(**data["paths"]),
        reprojection=ReprojectionConfig(**data["reprojection"]),
    )

    return _config_cache

def get_config() -> Config:
    """Get the cached config instance."""
    if _config_cache is None:
        return load_config()
    return _config_cache

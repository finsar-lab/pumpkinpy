import typer

from pumpkinpy.core import reprojection

app = typer.Typer(
    name="pumpkinpy",
    help="Postprocessing Utility for MintPy and Phase KINematics in Python",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False
)

@app.command()
def reproject(asc_filename: str, desc_filename: str) -> None:
    """Reprojects LOS as vertical andhorizontal components."""
    reprojection.run(asc_filename, desc_filename)


if __name__ == "__main__":
    app()

from pathlib import Path
import toml
import sparkml_base_classes


def test_versions_are_in_sync():
    """Checks if the pyproject.toml and package.__init__.py __version__ are in sync."""

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(str(path), encoding="utf-8") as file:
        pyproject = toml.loads(file.read())
    pyproject_version = pyproject["tool"]["poetry"]["version"]

    package_init_version = sparkml_base_classes.__version__

    assert (
        package_init_version == pyproject_version
    ), "Versions stated in __init__.py and in pypoject.toml are different"

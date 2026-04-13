from pathlib import Path
import tomllib


def test_pyproject_force_includes_bundled_atlas_niis_for_wheel_and_sdist():
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())

    expected_wheel = {
        "src/grouvox/atlases/aal.nii.gz": "grouvox/atlases/aal.nii.gz",
        "src/grouvox/atlases/ho_cort.nii.gz": "grouvox/atlases/ho_cort.nii.gz",
        "src/grouvox/atlases/ho_sub.nii.gz": "grouvox/atlases/ho_sub.nii.gz",
    }
    expected_sdist = {
        "src/grouvox/atlases/aal.nii.gz": "src/grouvox/atlases/aal.nii.gz",
        "src/grouvox/atlases/ho_cort.nii.gz": "src/grouvox/atlases/ho_cort.nii.gz",
        "src/grouvox/atlases/ho_sub.nii.gz": "src/grouvox/atlases/ho_sub.nii.gz",
    }

    targets = pyproject["tool"]["hatch"]["build"]["targets"]
    assert targets["wheel"]["force-include"] == expected_wheel
    assert targets["sdist"]["force-include"] == expected_sdist

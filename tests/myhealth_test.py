from myhealth.myHealth import app
from typer.testing import CliRunner


def test_get_nutrition_from_photo():
    expected = {
        "exit_code": 0,
        "stdouts": ["This food is a ", "with a ", " confidence."],
    }

    runner = CliRunner()
    got = runner.invoke(app, ["model/demo_images/hotdog.jpeg"])

    assert got.exit_code == expected["exit_code"]
    for stdout in expected["stdouts"]:
        assert stdout in got.stdout

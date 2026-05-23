from miniadk import run_cli
from miniadk.presets import coder


def main() -> None:
    run_cli(coder("."))


if __name__ == "__main__":
    main()

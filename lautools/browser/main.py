"""
pyQt6 GUI for Lautools
"""

import argparse
from pathlib import Path

from platformdirs import user_data_dir
from PySide6.QtWidgets import QApplication

from db import LaupyDB
from browser_window import BrowserWindow


APP_NAME = "laupy Browser"
DEFAULT_DATABASE_NAME = "laupy_beamtimes.sqlite"


def get_database_path(appdb: str | None) -> Path:
    """Determine which SQLite database the application should use."""
    if appdb:
        return Path(appdb).expanduser().resolve()
    data_dir = Path(user_data_dir(APP_NAME))
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir / DEFAULT_DATABASE_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LauTools beamtime manager"
    )
    parser.add_argument(
        "--laupy-db",
        metavar="PATH",
        help=(
            "Path to the application SQLite database. "
            f"Default: platform data directory/{DEFAULT_DATABASE_NAME}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_path = get_database_path(args.laupy_db)
    print(f"Using database: {database_path}")
    app = QApplication([])
    db = LaupyDB(database_path)
    window = BrowserWindow(db)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

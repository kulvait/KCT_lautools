"""
pyQt6 GUI for Lautools
"""

import argparse
import os
import signal
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
    parser.add_argument(
        "--qt-scale-factor",
        metavar="FACTOR",
        help="Override Qt scale factor, e.g. 1, 1.25, 1.5, 2",
    )
    parser.add_argument(
        "--no-qt-use-physical-dpi",
        action="store_true",
        help="Do not force QT_USE_PHYSICAL_DPI=1",
    )
    parser.add_argument(
        "--qt-disable-highdpi",
        action="store_true",
        help="Disable Qt high-DPI scaling where supported",
    )
    parser.add_argument(
        "--qt-scale-rounding-policy",
        choices=["Round", "PassThrough"],
        help="Qt scale factor rounding policy",
    )
    return parser.parse_args()


def configure_qt_scaling(args: argparse.Namespace) -> None:
    # Default for this application: prefer physical DPI.
    if not args.no_qt_use_physical_dpi:
        os.environ["QT_USE_PHYSICAL_DPI"] = "1"
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        os.environ["QT_SCALE_FACTOR"] = "1.9"

    if args.qt_scale_factor:
        os.environ["QT_SCALE_FACTOR"] = args.qt_scale_factor

    if args.qt_disable_highdpi:
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

    if args.qt_scale_rounding_policy:
        os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = (
            args.qt_scale_rounding_policy
        )


def main() -> None:
    args = parse_args()
    database_path = get_database_path(args.laupy_db)
    print(f"Using database: {database_path}")

    configure_qt_scaling(args)

    # Allow Ctrl+C from the launching terminal to terminate the GUI.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication([])
    db = LaupyDB(database_path)
    window = BrowserWindow(db)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

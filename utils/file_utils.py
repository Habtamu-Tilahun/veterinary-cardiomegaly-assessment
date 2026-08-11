from pathlib import Path
from werkzeug.utils import secure_filename


def save_file(file, upload_dir: Path) -> Path:
    """
    Save an uploaded file using a sanitized filename.

    Returns:
        Path: Location of the saved file.
    """

    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(file.filename)

    path = upload_dir / filename

    file.save(path)

    return path
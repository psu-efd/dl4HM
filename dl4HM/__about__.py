from sys import version_info

try:
    # Python 3.8+
    from importlib import metadata
except ImportError:
    try:
        import importlib_metadata as metadata
    except ImportError:
        __version__ = "unknown"

try:
    __version__ = metadata.version("dl4HM")
except Exception:
    __version__ = "unknown"


def get_dl4HM_version_info():
    return ", ".join(
        [
            f"dl4HM v{__version__}",
            f"Python {version_info.major}.{version_info.minor}.{version_info.micro}",
            "Copyright (c) 2021 Xiaofeng Liu"
        ]
    )

"""Config loading and project-root resolution for the pipeline drivers.

Relative paths inside a config (``model_file``, ``mapping_csv``, ...) are
resolved against the *project root of the config file*, not against the
sigmaflux installation. That is what allows a downstream project — e.g.
mammalian_flux_atlas — to keep its own configs, models and data in its own
repository and still drive the sigmaflux pipeline.

Resolution order for the root:

1. an explicit ``--root`` on the command line,
2. a ``root:`` key in the config (itself resolved relative to the config file),
3. the nearest ancestor of the config file containing ``pyproject.toml``,
   ``.git`` or ``setup.py``,
4. the directory containing the config file.
"""

import os
from pathlib import Path

import yaml

_ROOT_MARKERS = ("pyproject.toml", ".git", "setup.py")


def find_project_root(start):
    """Walk up from *start* to the nearest directory holding a root marker."""
    start = Path(start).resolve()
    candidates = [start] if start.is_dir() else []
    candidates += list(start.parents)
    for parent in candidates:
        if any((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
    return start if start.is_dir() else start.parent


def load_config(config_path, root=None):
    """Load a YAML config and determine the root its relative paths hang off.

    Args:
        config_path: path to the YAML config file.
        root: explicit project root override; otherwise resolved from the config
            or the config file's ancestor directories.

    Returns:
        (cfg, root) where root is an absolute :class:`str`.
    """
    config_path = Path(config_path).resolve()
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    if root is not None:
        resolved = Path(root)
    elif cfg.get("root"):
        resolved = Path(cfg["root"])
    else:
        resolved = find_project_root(config_path)

    if not resolved.is_absolute():
        resolved = (config_path.parent / resolved).resolve()

    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(resolved)
    return cfg, str(resolved)


def resolve_path(path, root):
    """Resolve *path* against *root*, leaving absolute paths untouched.

    Args:
        path: path to resolve, possibly relative; None passes through unchanged.
        root: base directory relative paths are resolved against.
    """
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(root, path)


def add_root_argument(parser):
    """Add the shared ``--root`` option to an argparse parser."""
    parser.add_argument(
        "--root",
        default=None,
        help="Project root that relative config paths resolve against "
             "(default: nearest pyproject.toml/.git above the config file)",
    )
    return parser


def default_output_dir(cfg, override=None):
    """Run directory for a config: ``<name>_YYYYMMDD`` unless overridden.

    Args:
        cfg: config dict, used for its 'name' when not overridden.
        override: explicit output directory; takes precedence over the default.
    """
    if override:
        return override
    import datetime
    return f"{cfg['name']}_{datetime.date.today().strftime('%Y%m%d')}"

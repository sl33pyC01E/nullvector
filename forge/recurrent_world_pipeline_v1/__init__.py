from .contract import DEFAULT_RELEASE
from .runtime import RecurrentWorldPipeline


def build_release(*args, **kwargs):
    from .release import build_release as implementation
    return implementation(*args, **kwargs)


def validate_release(*args, **kwargs):
    from .release import validate_release as implementation
    return implementation(*args, **kwargs)


__all__ = ["DEFAULT_RELEASE", "RecurrentWorldPipeline", "build_release", "validate_release"]

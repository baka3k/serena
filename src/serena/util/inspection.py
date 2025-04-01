from collections.abc import Generator
from typing import TypeVar

T = TypeVar("T")


def iter_subclasses(cls: type[T], recursive: bool = True) -> Generator[type[T], None, None]:
    """Iterate over all subclasses of a class. If recursive is True, also iterate over all subclasses of all subclasses."""
    for subclass in cls.__subclasses__():
        yield subclass
        if recursive:
            yield from iter_subclasses(subclass, recursive)

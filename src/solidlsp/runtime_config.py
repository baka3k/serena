"""Runtime dependency configuration loader for language servers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Optional

from .language_servers.common import RuntimeDependency


class RuntimeDependencyLoader:
    """Utility helpers to merge runtime dependency overrides."""

    @staticmethod
    def load_dependencies(
        base_dependencies: Iterable[RuntimeDependency],
        override_entries: Optional[Iterable[Mapping[str, object]]] = None,
    ) -> list[RuntimeDependency]:
        """Merge *override_entries* into *base_dependencies*.

        Entries are matched by the tuple ``(id, platform_id)``. If an override provides only a
        subset of the ``RuntimeDependency`` fields, the remaining values will fall back to the
        corresponding base dependency.
        """
        # Ensure we work on a copy in order to keep the base definition immutable for callers
        merged: dict[tuple[str | None, str | None], RuntimeDependency] = {(dep.id, dep.platform_id): dep for dep in base_dependencies}

        if not override_entries:
            return list(merged.values())

        for entry in override_entries:
            entry_id = entry.get("id")
            if entry_id is None:
                raise ValueError("Runtime dependency override requires an 'id' value")

            key = (entry_id, entry.get("platform_id"))
            base_dep = merged.get(key)

            if base_dep is not None:
                dep_data = asdict(base_dep)
                dep_data.update(entry)
            else:
                dep_data = dict(entry)

            merged[key] = RuntimeDependency(**dep_data)

        return list(merged.values())

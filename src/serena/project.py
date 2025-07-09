import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from serena.config.serena_config import ProjectConfig
from solidlsp.ls_config import Language


@dataclass
class Project:
    project_root: str
    project_config: ProjectConfig

    @property
    def project_name(self) -> str:
        return self.project_config.project_name

    @property
    def language(self) -> Language:
        return self.project_config.language

    @classmethod
    def load(cls, project_root: str | Path, autogenerate: bool = True) -> Self:
        project_root = Path(project_root).resolve()
        if not project_root.exists():
            raise FileNotFoundError(f"Project root not found: {project_root}")
        project_config = ProjectConfig.load(project_root, autogenerate=autogenerate)
        return cls(project_root=str(project_root), project_config=project_config)

    def path_to_project_yml(self) -> str:
        return os.path.join(self.project_root, self.project_config.rel_path_to_project_yml())

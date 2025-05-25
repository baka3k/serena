from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


def create_YAML(preserve_comments: bool = False) -> YAML:
    """
    Creates a YAML that can load/save with comments if preserve_comments is True.
    """
    typ = None if preserve_comments else "safe"
    result = YAML(typ=typ)
    result.preserve_quotes = preserve_comments
    return result


def load_yaml(path: str, preserve_comments: bool = False) -> dict:
    with open(path, encoding="utf-8") as f:
        yaml = create_YAML(preserve_comments)
        return yaml.load(f)


def save_yaml(path: str, data: dict | CommentedMap, preserve_comments: bool = False) -> None:
    yaml = create_YAML(preserve_comments)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

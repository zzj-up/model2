"""Node-level sequence/structure ranking architecture (independent of legacy DTI)."""

__all__ = ["CONFIG", "NodeRanker"]


def __getattr__(name):
    # Feature preparation must not import the task model/PyG unnecessarily.
    if name in __all__:
        from .model import CONFIG, NodeRanker
        globals().update(CONFIG=CONFIG, NodeRanker=NodeRanker)
        return globals()[name]
    raise AttributeError(name)

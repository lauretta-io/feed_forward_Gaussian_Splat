import warnings


_WEAVE_INITIALIZED = False


def init_weave(project: str | None, enabled: bool = True) -> bool:
    global _WEAVE_INITIALIZED
    if not enabled or project is None or project == "":
        return False
    if _WEAVE_INITIALIZED:
        return True

    try:
        import weave

        weave.init(project)
        _WEAVE_INITIALIZED = True
        return True
    except ImportError:
        warnings.warn("Weave logging requested, but `weave` is not installed.")
    except Exception as e:
        warnings.warn(f"Failed to initialize Weave project {project!r}: {e}")
    return False


def finish_weave() -> None:
    global _WEAVE_INITIALIZED
    if not _WEAVE_INITIALIZED:
        return

    try:
        import weave

        weave.finish()
    except Exception as e:
        warnings.warn(f"Failed to finish Weave run: {e}")
    finally:
        _WEAVE_INITIALIZED = False

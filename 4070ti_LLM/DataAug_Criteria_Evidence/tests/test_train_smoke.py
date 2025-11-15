def test_smoke():
    # Minimal smoke test just ensures CLI loads; training is project-specific
    import importlib

    cli = importlib.import_module("psy_agents_noaug.cli")
    assert hasattr(cli, "main")

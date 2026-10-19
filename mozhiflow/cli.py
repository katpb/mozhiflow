"""MozhiFlow console entry point."""

def main(argv=None):
    from .pipeline import main as run
    return run(argv)

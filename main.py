import sys


# With no arguments the app opens. With a command (run, check, modules, --help or --version)
# Coefflow runs from the command line instead, without loading the window (see engine/cli.py)
COMMANDS = {"run", "check", "modules", "-h", "--help", "--version"}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        from engine.cli import main
        sys.exit(main(sys.argv[1:]))

    from ui.app import run
    run()

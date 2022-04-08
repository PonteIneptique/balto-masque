import os
from typing import Iterable
import click


@click.command()
@click.argument("path", type=click.Path(file_okay=True, exists=True, dir_okay=False), nargs=-1)
@click.option("--port", default=8888, help="port on which to run the app")
def run(path: Iterable[str], port: int):
    os.environ["XML"] = "|".join(path)
    from baltomasque.app import app
    app.run(port=port)


if __name__ == "__main__":
    run()

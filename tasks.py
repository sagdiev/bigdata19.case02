from invoke import task
import sys


@task
def init(c):
    """Initialize cloned project."""
    c.run(f'{sys.executable} -m pip install -r requirements.txt')


@task
def run(c, path):
    """Run python script."""
    c.run(f'{sys.executable} {path}')

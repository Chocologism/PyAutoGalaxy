from os.path import dirname, realpath

import pytest
from matplotlib import pyplot
from os import path
from autoconf import conf


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path(request):
    if dirname(realpath(__file__)) in str(request.module):
        conf.instance = conf.Config(
            path.join(directory, "unit/config"), path.join(directory, "unit/output")
        )

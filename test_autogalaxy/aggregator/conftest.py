import pytest
from os import path
import os
import shutil

from autoconf import conf
import autofit as af
import autogalaxy as ag
from autofit.non_linear.samples import Sample


@pytest.fixture(autouse=True)
def set_test_mode():
    os.environ["PYAUTOFIT_TEST_MODE"] = "1"
    yield
    del os.environ["PYAUTOFIT_TEST_MODE"]


def clean(database_file):
    database_sqlite = path.join(conf.instance.output_path, f"{database_file}.sqlite")

    if path.exists(database_sqlite):
        os.remove(database_sqlite)

    result_path = path.join(conf.instance.output_path, database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


def aggregator_from(database_file, analysis, model, samples):
    result_path = path.join(conf.instance.output_path, database_file)

    clean(database_file=database_file)

    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=database_file)
    search.fit(model=model, analysis=analysis)

    database_file = path.join(conf.instance.output_path, f"{database_file}.sqlite")

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    return agg


@pytest.fixture(name="model")
def make_model():
    return af.Collection(
        galaxies=af.Collection(
            g0=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.Sersic),
            g1=af.Model(ag.Galaxy, redshift=1.0, light=ag.lp.Sersic),
        )
    )


@pytest.fixture(name="samples")
def make_samples(model):
    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(centre=(0.0, 1.0)))
    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.Sersic())

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return ag.m.MockSamples(
        model=model,
        sample_list=sample_list,
        max_log_likelihood_instance=plane,
        gaussian_tuples=[(1.0, 2.0)] * model.prior_count,
    )

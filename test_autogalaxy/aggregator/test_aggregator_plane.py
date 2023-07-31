from os import path

from autoconf import conf
import autofit as af
import autogalaxy as ag


from test_autogalaxy.aggregator.conftest import clean


def test__plane_randomly_drawn_via_pdf_gen_from(masked_imaging_7x7, adapt_model_image_7x7, adapt_galaxy_image_path_dict_7x7, samples, model):
    path_prefix = "aggregator_plane_gen"

    database_file = path.join(conf.instance.output_path, "plane.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    analysis.adapt_model_image = adapt_model_image_7x7
    analysis.adapt_galaxy_image_path_dict = adapt_galaxy_image_path_dict_7x7

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    plane_agg = ag.agg.PlaneAgg(aggregator=agg)
    plane_pdf_gen = plane_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for plane_gen in plane_pdf_gen:
        for plane in plane_gen:
            i += 1

            assert plane.galaxies[0].redshift == 0.5
            assert plane.galaxies[0].light.centre == (10.0, 10.0)
            assert plane.galaxies[1].redshift == 1.0


            assert (plane.galaxies[0].adapt_model_image == adapt_model_image_7x7).all()
            assert (plane.galaxies[0].adapt_galaxy_image == adapt_galaxy_image_path_dict_7x7["('galaxies', 'g0')"]).all()
            assert (plane.galaxies[1].adapt_galaxy_image == adapt_galaxy_image_path_dict_7x7["('galaxies', 'g1')"]).all()

    assert i == 2

    clean(database_file=database_file, result_path=result_path)
#
#
# def test__plane_all_above_weight_gen(masked_imaging_7x7, samples, model):
#     path_prefix = "aggregator_plane_gen"
#
#     database_file = path.join(conf.instance.output_path, "plane.sqlite")
#     result_path = path.join(conf.instance.output_path, path_prefix)
#
#     clean(database_file=database_file, result_path=result_path)
#
#     search = ag.m.MockSearch(
#         samples=samples, result=ag.m.MockResult(model=model, samples=samples)
#     )
#     search.paths = af.DirectoryPaths(path_prefix=path_prefix)
#     analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)
#     search.fit(model=model, analysis=analysis)
#
#     agg = af.Aggregator.from_database(filename=database_file)
#     agg.add_directory(directory=result_path)
#
#     plane_agg = ag.agg.PlaneAgg(aggregator=agg)
#     plane_pdf_gen = plane_agg.all_above_weight_gen_from(minimum_weight=-1.0)
#     weight_pdf_gen = plane_agg.weights_above_gen_from(minimum_weight=-1.0)
#
#     i = 0
#
#     for plane_gen, weight_gen in zip(plane_pdf_gen, weight_pdf_gen):
#         for plane in plane_gen:
#             i += 1
#
#             if i == 1:
#                 assert plane.galaxies[0].redshift == 0.5
#                 assert plane.galaxies[0].light.centre == (1.0, 1.0)
#                 assert plane.galaxies[1].redshift == 1.0
#
#             if i == 2:
#                 assert plane.galaxies[0].redshift == 0.5
#                 assert plane.galaxies[0].light.centre == (10.0, 10.0)
#                 assert plane.galaxies[1].redshift == 1.0
#
#         for weight in weight_gen:
#             if i == 0:
#                 assert weight == 0.0
#
#             if i == 1:
#                 assert weight == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file, result_path=result_path)
#
#
# def test__results_array_froms_file(path):
#
#     results = [
#         MockResult(log_likelihood=1.0),
#         MockResult(log_likelihood=2.0),
#         MockResult(log_likelihood=3.0),
#         MockResult(log_likelihood=4.0),
#     ]
#
#     lower_limit_lists = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]
#     physical_lower_limits_lists = [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]
#
#     grid_search_result = af.GridSearchResult(
#         results=results,
#         physical_lower_limits_lists=physical_lower_limits_lists,
#         lower_limit_lists=lower_limit_lists,
#     )
#
#     aggregator = MockAggregator(grid_search_result=grid_search_result)
#
#     array = ag.agg.grid_search_result_as_array(aggregator=aggregator)
#
#     assert array.native == pytest.approx(np.array([[3.0, 2.0], [1.0, 4.0]]), 1.0e4)
#     assert array.pixel_scales == (1.0, 1.0)
#
#
# def test__results_array_from_real_grid_search_pickle(path):
#
#     with open("{}/{}.pickle".format(path, "grid_search_result"), "rb") as f:
#         grid_search_result = pickle.load(f)
#
#     array = ag.agg.grid_search_log_evidences_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array.native)
#
#     array = ag.agg.grid_search_subhalo_masses_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array.native)
#
#     array = ag.agg.grid_search_subhalo_centres_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array)

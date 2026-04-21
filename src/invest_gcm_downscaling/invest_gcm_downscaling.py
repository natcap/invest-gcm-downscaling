"""
This work is an adaptation of the GCMClimTool Library
by Angarita H., Yates D., Depsky N. 2014-2021
"""

from datetime import datetime
import logging
import gcsfs
import os
from osgeo import ogr
from pprint import pformat
import pandas
import warnings
import xarray

from knn import knn
from knn import plot


from natcap.invest import spec
from natcap.invest import validation
from natcap.invest.unit_registry import u
from natcap.invest import gettext

LOGGER = logging.getLogger(__name__)
LOG_FMT = (
    "%(asctime)s "
    "(%(name)s) "
    "%(module)s.%(funcName)s(%(lineno)d) "
    "%(levelname)s %(message)s")

DATE_EXPR = r"^(18|19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
LIST_MODELS = [spec.Option(key=k) for k in knn.MODEL_LIST]

_model_description = gettext(
    """
    The InVEST plugin for GCM Downscaling (https://github.com/natcap/gcm-downscaling)
    is used to downscale gridded precipitation data from CMIP6 (Coupled Model
    Intercomparison Project) GCMs (General Circulation Models) using observed
    historical precipitation patterns. The goal is to produce realistic future
    daily precipitation data that reflect both climate model projections and
    statistical patterns of observed historical records. The model supports
    both hindcasts and future projections.
    """)

MODEL_SPEC = spec.ModelSpec(
    model_id='gcm_downscaling',
    model_title=gettext('GCM Downscaling'),
    userguide='https://github.com/natcap/invest-gcm-downscaling/blob/main/README.md',
    reporter='',
    about=_model_description,
    module_name=__name__,
    input_field_order=[
        ['workspace_dir', 'aoi_path'],
        ['reference_period_start_date', 'reference_period_end_date'],
        ['prediction_start_date', 'prediction_end_date'],
        ['hindcast'],
        ['gcm_model'],
        ['upper_precip_percentile', 'lower_precip_threshold'],
        ['observed_dataset_path']],
    inputs=[
        spec.WORKSPACE,
        spec.N_WORKERS,
        spec.VectorInput(
            id='aoi_path',
            name='Area of Interest',
            about=gettext(
                'Path to a GDAL polygon vector representing the Area of Interest '
                '(AOI). Coordinates represented by longitude, latitude decimal degrees '
                '(e.g. WGS84).'),
            required=True,
            fields=[],
            geometry_types={'POLYGON', 'MULTIPOLYGON'},
            projected=False
        ),
        spec.StringInput(
            id='reference_period_start_date',
            name='Reference Period Start Date',
            about=gettext(
                'First day in the reference period, which is used to calculate '
                'climate "normals". The reference period should typically '
                'span about 30 years or more. If ``observed_dataset`` is not '
                'input, reference period must extend past 1979, and should '
                'extend at least 30 years beyond 1979. Format: "YYYY-MM-DD"'),
            required=True,
            regexp=DATE_EXPR
        ),
        spec.StringInput(
            id='reference_period_end_date',
            name='Reference Period End Date',
            about=gettext(
                'Last day in the reference period, which is used to calculate '
                'climate "normals". The reference period should typically '
                'span about 30 years or more. If ``observed_dataset`` is not '
                'input, reference period must extend past 1979, and should '
                'extend at least 30 years beyond 1979. Format: "YYYY-MM-DD"'),
            required=True,
            regexp=DATE_EXPR
        ),
        spec.StringInput(
            id='prediction_start_date',
            name='Prediction Start Date',
            about=gettext("First day in the simulation period, in format 'YYYY-MM-DD'"),
            required='gcm_model',
            regexp=DATE_EXPR
        ),
        spec.StringInput(
            id='prediction_end_date',
            name='Prediction End Date',
            about=gettext("Last day in the simulation period, in format 'YYYY-MM-DD'"),
            required='gcm_model',
            regexp=DATE_EXPR
        ),
        spec.BooleanInput(
            id='hindcast',
            name='Hindcast',
            about=gettext(
                'If True, observed data (MSWEP) is substituted for GCM '
                'data and the prediction period is set to match the date '
                'range of the observed dataset.'), #{knn.MSWEP_DATE_RANGE}
            required=True
        ),
        spec.OptionStringInput(
            id='gcm_model',
            name='GCM Model',
            about=gettext(
                "A CMIP6 (Coupled Model Intercomparison Project Phase 6) "
                "climate model code. These models are used to simulate past, "
                "present, and future climate conditions. Each model represents "
                "the Earth's climate system using different assumptions, "
                "physics, and resolutions. Each model will be used to "
                "generate a single downscaled product for each CMIP6 Shared "
                "Socioeconomic Pathways (SSP) experiment."),
            options=[spec.Option(key="")] + LIST_MODELS,
            required='not hindcast'
        ),
        spec.PercentInput(
            id='upper_precip_percentile',
            name='Upper Precipitation Percentile',
            about=gettext(
                'A percentile (from 0-100) with which to extract the '
                'absolute precipitation value that will be the upper '
                'boundary (inclusive) of the middle bin of precipitation '
                'states.'),
            required=True,
            expression="(value >= 0) & (value <= 100)",
        ),
        spec.NumberInput(
            id='lower_precip_threshold',
            name='Lower Precipitation Threshold',
            about=gettext(
                'The lower boundary of the middle bin of precipitation states'),
            required=True,
            units=u.millimeter,
            expression="(value >= 0)",
        ),
        spec.SingleBandRasterInput(
            id='observed_dataset_path',
            name='Observed Dataset',
            about=gettext(
                'If provided, this dataset will be used instead of MSWEP '
                'as the source of observed, historical preciptation. The '
                'dataset should be a netCDF or other xarray.open_dataset '
                'readable format. It should contain coordinates and '
                'variables named & defined as: '
                'lat - decimal degrees (-90 : 90), '
                'lon - decimal degrees (-180 : 180) or (0 : 360), '
                'time - daily timesteps in units that can be parsed to numpy.datetime64'),
            required=False,
            units=u.millimeter
            )
    ],
    outputs=[
        spec.SingleBandRasterOutput(
            id='downscaled_precip_[MODEL]_[EXPERIMENT].nc',
            path='output/downscaled_precip_[MODEL]_[EXPERIMENT].nc',
            about=gettext(
                'Gridded NetCDF file containing the downscaled daily '
                'precipitation time series for the specified climate '
                'model and experiment scenario.'),
            bands=[]
        ),
        spec.FileOutput(
            id='downscaled_precip_[MODEL]_[EXPERIMENT].pdf',
            path='output/downscaled_precip_[MODEL]_[EXPERIMENT].pdf',
            about=gettext(
                'Report with graphs and visualizations of downscaled '
                'precipitation data for specified model and experiment')
        ),
        spec.SingleBandRasterOutput(
            id='downscaled_precip_hindcast.nc',
            path='output/downscaled_precip_hindcast.nc',
            about=gettext(
                'Gridded NetCDF file with downscaled historical '
                'precipitation data (hindcast), serving as a baseline for '
                'model validation.'),
            bands=[]
        ),
        spec.FileOutput(
            id='downscaled_precip_hindcast.pdf',
            path='output/downscaled_precip_hindcast.pdf',
            about=gettext(
                'Report with graphs and visualizations of downscaled '
                'hindcast precipitation data.')
        ),
        spec.SingleBandRasterOutput(
            id='aoi_mask_[MODEL].nc',
            path='intermediate/aoi_mask_[MODEL].nc',
            about=gettext('Area of Interest (AOI) mask'),
            units=u.none
        ),
        spec.CSVOutput(
            id='bootstrapped_dates_precip_[MODEL]_[EXPERIMENT].csv',  # EXPERIMENT can also be 'hindcast'
            path='intermediate/bootstrapped_dates_precip_[MODEL]_[EXPERIMENT].csv',
            about=gettext(
                'Bootstrapped dates and associated precipitation '
                'values used in the downscaling process.'),
            columns=[
                spec.StringOutput(
                    id='historic_date',
                    about=gettext(
                        'Date from the historical record used '
                        'in bootstrapping.')
                ),
                spec.NumberOutput(
                    id='historic_precip',
                    about=gettext('Historic precipitation'),
                    units=u.millimeter
                ),
                spec.IntegerOutput(
                    id='wet_state',
                    about=gettext(
                        'Dry/wet/very wet state classification for '
                        'the historic date.')
                ),
                spec.IntegerOutput(
                    id='next_wet_state',
                    about=gettext(
                        'Predicted dry/wet/very wet state for the '
                        'subsequent time step.')
                ),
                spec.StringOutput(
                    id='next_historic_date',
                    about='Next date in the bootstrapped sequence.'
                )
            ]
        ),
        spec.RasterOutput(
            id='extracted_[MODEL]_[EXPERIMENT].nc',  # EXPERIMENT can also be 'hindcast'
            path='intermediate/extracted_[MODEL]_[EXPERIMENT].nc',
            about=gettext(
                'NetCDF file containing precipitation data extracted from the '
                'specified model and experiment (or hindcast), prior to'
                'downscaling.'),
            bands=[]
        ),
        spec.RasterOutput(
            id='aoi_mask_mswep.nc',
            path='intermediate/aoi_mask_mswep.nc',
            about=gettext(
                'AOI mask NetCDF file.'),
            bands=[]
        ),
        spec.RasterOutput(
            id='extracted_mswep.nc',
            path='intermediate/extracted_mswep.nc',
            about=gettext(
                'NetCDF file with precipitation data extracted from the '
                'MSWEP dataset, used as observational reference.'),
            bands=[]
        ),
        spec.RasterOutput(
            id='mswep_mean.nc',
            path='intermediate/mswep_mean.nc',
            about=gettext(
                'NetCDF file representing the mean precipitation from '
                'the MSWEP dataset over the analysis period.'),
            bands=[]
        ),
        spec.RasterOutput(
            id='extracted_[MODEL]_historical.nc',
            path='intermediate/extracted_[MODEL]_historical.nc',
            about=gettext(
                'NetCDF file with historical precipitation data extracted '
                'from the [model].'),
            bands=[]
        ),
        spec.CSVOutput(
            id='bootstrapped_dates_precip_hindcast.csv',
            path='intermediate/bootstrapped_dates_precip_hindcast.csv',
            about=gettext(
                'Bootstrapped dates precipitation hindcast table.'),
            bands=[]
        ),
        spec.RasterOutput(
            id='pr_day_[MODEL]_[EXPERIMENT]_mean.nc',
            path='intermediate/pr_day_[MODEL]_[EXPERIMENT]_mean.nc',
            about=gettext(
                'NetCDF file containing the daily mean precipitation '
                'values for the specified model and experiment.'),
            bands=[]
        ),
        spec.CSVOutput(
            id='synthesized_extreme_precip_[MODEL]_[EXPERIMENT].csv',
            path='intermediate/synthesized_extreme_precip_[MODEL]_[EXPERIMENT].csv',
            about=gettext(
                'CSV file summarizing synthesized extreme precipitation '
                'events for the specified model and experiment.'),
            columns=[
                spec.NumberOutput(
                    id='historic_sample',
                    about=gettext(
                        'Precipitation value from the historical sample'),
                    units=u.millimeter
                ),
                spec.NumberOutput(
                    id='forecast_sample',
                    about=gettext(
                        'Projected precipitation value from the '
                        'forecast sample'),
                    units=u.millimeter
                )
            ]
        ),
        spec.TASKGRAPH_CACHE
    ],
)


def _check_lonlat_coords(vector_path):
    ds = ogr.Open(vector_path)
    layer = ds.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref is None:
        raise ValueError("AOI vector file has no spatial reference system defined.")

    if not spatial_ref.IsGeographic():
        raise ValueError(
            "The AOI vector file must use geographic coordinates (longitude "
            "and latitude in decimal degrees), such as WGS 84 (EPSG:4326). "
            "However, a projected coordinate system was found instead. To "
            "fix this, reproject your vector data to EPSG:4326 (or similar)."
        )


def execute(args):
    """Create a downscaled precipitation product for an area of interest.

    Args:
        args['aoi_path'] (str): a path to a GDAL polygon vector. Coordinates
            represented by longitude, latitude decimal degrees (e.g. WGS84).
        args['workspace_dir'] (str): a path to the directory where this program
            writes output and other temporary files.
        args['reference_period_start_date'] (string): ('YYYY-MM-DD')
            first day in the reference period, which is used to
            calculate climate "normals".
        args['reference_period_end_date'] (string): ('YYYY-MM-DD')
            last day in the reference period, which is used to
            calculate climate "normals".
        args['lower_precip_threshold'] (float): the lower boundary of the
            middle bin of precipitation states. Units: mm
        args['upper_precip_percentile'] (float): a percentile (from 0:100) with
            which to extract the absolute precipitation value that will be the
            upper boundary (inclusive) of the middle bin of precipitation states.
        args['hindcast'] (bool): If True, observed data (MSWEP) is substituted
            for GCM data and the prediction period is set to match the date
            range of the observed dataset (``knn.MSWEP_DATE_RANGE``).
        args['prediction_start_date'] (string, optional):
            ('YYYY-MM-DD') first day in the simulation period.
            Required if `hindcast=False`.
        args['prediction_end_date'] (string, optional):
            ('YYYY-MM-DD') last day in the simulation period.
            Required if `hindcast=False`.
        args['gcm_model'] (string, optional): a string representing a CMIP6 model code.
            Each model will be used to generate a single downscaled product for
            each experiment. Available models are stored in ``knn.MODEL_LIST``.
            Required if `hindcast=False`.
        args['observed_dataset_path'] (string, optional): if provided, this
            dataset will be used instead of MSWEP as the source of observed,
            historical preciptation. The dataset should be a netCDF or other
            ``xarray.open_dataset`` readable format. It should contain
            coordinates and variables named & defined as,

                Coordinates:
                * ``lat``  - decimal degrees (-90 : 90)
                * ``lon``  - decimal degrees (-180 : 180) or (0 : 360)
                * ``time`` - daily timesteps in units that can be parsed to
                             ``numpy.datetime64``

                Variables:
                * ``precipitation`` - dimensions: (time, lat, lon)
                                      units: millimeter

        args['n_workers'] (int, optional): The number of worker processes to
            use. If omitted, computation will take place in the current process.
            If a positive number, tasks can be parallelized across this many
            processes, which can be useful if `knn.GCM_EXPERIMENT_LIST` contain
            multiple items.
    """
    LOGGER.info(pformat(args))
    args, file_registry, graph = MODEL_SPEC.setup(args)

    # Check AOI spatial reference
    _check_lonlat_coords(args['aoi_path'])

    ref_end = pandas.to_datetime(args['reference_period_end_date'])
    ref_start = pandas.to_datetime(args['reference_period_start_date'])

    # check that end dates are after start dates
    if ref_end <= ref_start:
        raise ValueError('Reference end date must be after reference start date.')

    if args.get('prediction_start_date') and args.get('prediction_end_date'):
        if pandas.to_datetime(args['prediction_end_date']) <= pandas.to_datetime(
                args['prediction_start_date']):
            raise ValueError('Prediction end date must be after prediction start date.')

    # if length of reference period is less than 30 years, throw a Warning
    # if the reference period is too short, there will be a cryptic error
    # if ref_end - ref_start < pandas.datetime.timedelta(years=30):
    # issue with this is that we should actually check if overlap with
    # historical dataset is > 30 years
    if ref_end < ref_start + pandas.DateOffset(years=30):
        warnings.warn("The reference period is less than 30 years.",
                      category=UserWarning)

    # Validate reference dates if using MSWEP data
    if 'observed_dataset_path' not in args or \
            args['observed_dataset_path'] is None:
        min_mswep_date = pandas.to_datetime(knn.MSWEP_DATE_RANGE[0])
        max_mswep_date = pandas.to_datetime(knn.MSWEP_DATE_RANGE[1])
        if (ref_start > max_mswep_date or ref_end < min_mswep_date):
            raise ValueError(
                f'the requested reference time period is outside the '
                f'time-range of MSWEP ({min_mswep_date} : {max_mswep_date})'
            )

    prediction_dates = (args.get('prediction_start_date') or None,
                        args.get('prediction_end_date') or None)

    reference_period_dates = (args['reference_period_start_date'],
                              args['reference_period_end_date'])

    if args.get('gcm_model'):  # only add this model arg if gcm_model != ''
        gcm_model_list = [args['gcm_model'].upper()]
    else:
        gcm_model_list = []

    mswep_extract_path = file_registry['extracted_mswep.nc']
    aoi_mask_mswep_path = file_registry['aoi_mask_mswep.nc']
    mswep_netcdf_path = file_registry['mswep_mean.nc']

    rasterize_dependent_task_list = []
    if 'observed_dataset_path' in args and \
            args['observed_dataset_path'] is not None:
        mswep_extract_path = args['observed_dataset_path']
    else:
        extract_mswep_task = graph.add_task(
            func=knn.extract_from_zarr,
            kwargs={
                'zarr_path': knn.MSWEP_STORE_PATH,
                'aoi_path': args['aoi_path'],
                'target_path': mswep_extract_path,
                'open_chunks': knn.MSWEP_ZARR_CHUNKS,
            },
            task_name='Extract MSWEP data by bounding box',
            target_path_list=[mswep_extract_path],
            dependent_task_list=[]
        )
        rasterize_dependent_task_list.append(extract_mswep_task)
        graph.join()

    rasterize_aoi_mswep_task = graph.add_task(
        func=knn.rasterize_aoi,
        kwargs={
            'aoi_path': args['aoi_path'],
            'netcdf_path': mswep_extract_path,
            'target_filepath': aoi_mask_mswep_path,
        },
        task_name='Rasterize AOI onto the MSWEP grid.',
        target_path_list=[aoi_mask_mswep_path],
        dependent_task_list=rasterize_dependent_task_list
    )

    reduce_mswep_task = graph.add_task(
        func=knn.reduce_netcdf,
        kwargs={
            'source_file_list': [mswep_extract_path],
            'target_filepath': mswep_netcdf_path,
            'aoi_netcdf_path': aoi_mask_mswep_path
        },
        task_name='Reduce MSWEP to average value within AOI.',
        target_path_list=[mswep_netcdf_path],
        dependent_task_list=[rasterize_aoi_mswep_task]
    )

    if args['hindcast']:
        hindcast_target_csv_path = file_registry[
            'bootstrapped_dates_precip_hindcast.csv']
        hindcast_date_range = knn.MSWEP_DATE_RANGE
        if 'observed_dataset_path' in args and \
                args['observed_dataset_path'] is not None:
            with xarray.open_dataset(mswep_netcdf_path) as dataset:
                min_date = str(dataset.time.min().values)[:10]
                max_date = str(dataset.time.max().values)[:10]
            hindcast_date_range = (min_date, max_date)
        hind_bootstrap_dates_task = graph.add_task(
            func=knn.bootstrap_dates_precip,
            kwargs={
                'observed_data_path': mswep_netcdf_path,
                'prediction_dates': hindcast_date_range,
                'reference_period_dates': reference_period_dates,
                'lower_precip_threshold': args['lower_precip_threshold'],
                'upper_precip_percentile': args['upper_precip_percentile'],
                'target_csv_path': hindcast_target_csv_path,
                'hindcast': True
            },
            task_name='Bootstrap dates for precipitation',
            target_path_list=[hindcast_target_csv_path],
            dependent_task_list=[reduce_mswep_task]
        )
        hindcast_target_netcdf_path = file_registry['downscaled_precip_hindcast.nc']
        hind_downscale_precip_task = graph.add_task(
            func=knn.downscale_precip,
            kwargs={
                'bootstrapped_dates_path': hindcast_target_csv_path,
                'gridded_observed_precip': mswep_extract_path,
                'aoi_mask_path': aoi_mask_mswep_path,
                'target_netcdf_path': hindcast_target_netcdf_path
            },
            task_name='Downscale Precipitation',
            target_path_list=[hindcast_target_netcdf_path],
            dependent_task_list=[hind_bootstrap_dates_task]
        )
        hindcast_target_pdf_path = os.path.splitext(
            hindcast_target_netcdf_path)[0] + '.pdf'
        report_task = graph.add_task(
            func=plot.plot,
            kwargs={
                'dates_filepath': hindcast_target_csv_path,
                'precip_filepath': hindcast_target_netcdf_path,
                'observed_mean_precip_filepath': mswep_netcdf_path,
                'observed_precip_filepath': mswep_extract_path,
                'aoi_netcdf_path': aoi_mask_mswep_path,
                'reference_period_dates': reference_period_dates,
                'hindcast': True,
                'target_filename': hindcast_target_pdf_path
            },
            task_name='Report',
            target_path_list=[hindcast_target_pdf_path],
            dependent_task_list=[hind_downscale_precip_task]
        )

    gcs_filesystem = gcsfs.GCSFileSystem(token='anon')
    for gcm_model in gcm_model_list:
        historical_gcm_files = gcs_filesystem.glob(
            f"{knn.BUCKET}/{knn.GCM_PREFIX}/{gcm_model}/{knn.GCM_PRECIP_VAR}_day_{gcm_model}_historical_*.zarr")
        if len(historical_gcm_files) == 0:
            LOGGER.warning(
                f'No files found for model: {gcm_model}, experiment: historical; '
                f'skipping model {gcm_model}. {f"{knn.BUCKET}/{knn.GCM_PREFIX}/{gcm_model}/{knn.GCM_PRECIP_VAR}_day_{gcm_model}_historical_*.zarr"}')
            continue
        if len(historical_gcm_files) > 1:
            LOGGER.warning(
                f'Ambiguous files found for model: {gcm_model}, experiment: historical; '
                f'Found: {historical_gcm_files}; '
                f'skipping model {gcm_model}.')
            continue

        # validate that reference dates fall within range of historical data
        with xarray.open_dataset(
                f'{knn.GCS_PROTOCOL}{historical_gcm_files[0]}',
                decode_times=xarray.coders.CFDatetimeCoder(use_cftime=True),
                engine='zarr',
                backend_kwargs={"storage_options": {"token": 'anon'}}
                    ) as gcm_hist_dataset:
            knn.validate(gcm_hist_dataset, *reference_period_dates)
        # validate forecast dates fall within range of future data
        future_gcm_files = gcs_filesystem.glob(
                f"{knn.BUCKET}/{knn.GCM_PREFIX}/{gcm_model}/{knn.GCM_PRECIP_VAR}_day_{gcm_model}_ssp*.zarr")
        if len(future_gcm_files) == 0:
            LOGGER.warning(
                f'No files found for model: {gcm_model}. Skipping.')
            continue
        with xarray.open_dataset(
                f'{knn.GCS_PROTOCOL}{future_gcm_files[0]}',
                decode_times=xarray.coders.CFDatetimeCoder(use_cftime=True),
                engine='zarr',
                backend_kwargs={"storage_options": {"token": 'anon'}}
                ) as future_gcm_dataset:
            knn.validate(future_gcm_dataset, *prediction_dates)

        gcm_historical_extract_path = file_registry[
            ('extracted_[MODEL]_historical.nc', gcm_model)]
        extract_historical_gcm_task = graph.add_task(
            func=knn.extract_from_zarr,
            kwargs={
                'zarr_path': f'{knn.GCS_PROTOCOL}{historical_gcm_files[0]}',
                'aoi_path': args['aoi_path'],
                'target_path': gcm_historical_extract_path,
                'open_chunks': knn.CMIP_ZARR_CHUNKS
            },
            task_name='Extract GCM historical data by bounding box',
            target_path_list=[gcm_historical_extract_path],
            dependent_task_list=[]
        )

        aoi_mask_gcm_path = file_registry[('aoi_mask_[MODEL].nc', gcm_model)]
        rasterize_aoi_gcm_task = graph.add_task(
            func=knn.rasterize_aoi,
            kwargs={
                'aoi_path': args['aoi_path'],
                'netcdf_path': gcm_historical_extract_path,
                'target_filepath': aoi_mask_gcm_path,
            },
            task_name='Rasterize AOI onto the GCM grid.',
            target_path_list=[aoi_mask_gcm_path],
            dependent_task_list=[extract_historical_gcm_task]
        )
        for gcm_experiment in knn.GCM_EXPERIMENT_LIST:
            future_gcm_files = gcs_filesystem.glob(
                f"{knn.BUCKET}/{knn.GCM_PREFIX}/{gcm_model}/{knn.GCM_PRECIP_VAR}_day_{gcm_model}_{gcm_experiment}_*.zarr")

            if len(future_gcm_files) == 0:
                LOGGER.warning(
                    f'No files found for model: {gcm_model}, experiment: {gcm_experiment}'
                    f'skipping experment: {gcm_experiment} - {gcm_model}.')
                continue
            if len(future_gcm_files) > 1:
                LOGGER.warning(
                    f'Ambiguous files found for model: {gcm_model}, experiment: {gcm_experiment}'
                    f'Found: {future_gcm_files}'
                    f'skipping experiment: {gcm_experiment} - {gcm_model}.')
                continue
            LOGGER.info(f'Starting {gcm_model} {gcm_experiment}')

            target_csv_path = file_registry[
                ('bootstrapped_dates_precip_[MODEL]_[EXPERIMENT].csv',
                 gcm_model, gcm_experiment)]
            target_netcdf_path = file_registry[
                ('downscaled_precip_[MODEL]_[EXPERIMENT].nc',
                 gcm_model, gcm_experiment)]

            gcm_netcdf_path = file_registry[
                (f"{knn.GCM_PRECIP_VAR}_day_[MODEL]_[EXPERIMENT]_mean.nc",
                 gcm_model, gcm_experiment)]

            gcm_future_extract_path = file_registry[
                ("extracted_[MODEL]_[EXPERIMENT].nc",
                 gcm_model, gcm_experiment)]

            extract_future_gcm_task = graph.add_task(
                func=knn.extract_from_zarr,
                kwargs={
                    'zarr_path': f'{knn.GCS_PROTOCOL}{future_gcm_files[0]}',
                    'aoi_path': args['aoi_path'],
                    'target_path': gcm_future_extract_path,
                    'open_chunks': knn.CMIP_ZARR_CHUNKS
                },
                task_name='Extract GCM future data by bounding box',
                target_path_list=[gcm_future_extract_path],
                dependent_task_list=[]
            )

            target_extreme_values_path = file_registry[
                ('synthesized_extreme_precip_[MODEL]_[EXPERIMENT].csv',
                 gcm_model, gcm_experiment)
            ]

            extreme_values_task = graph.add_task(
                func=knn.synthesize_extreme_values,
                kwargs={
                    'historical_gcm_path': gcm_historical_extract_path,
                    'reference_period_dates': reference_period_dates,
                    'forecast_gcm_path': gcm_future_extract_path,
                    'prediction_period_dates': prediction_dates,
                    'target_csv_path': target_extreme_values_path,
                },
                task_name='Synthesize extreme values',
                target_path_list=[target_extreme_values_path],
                store_result=True,
                dependent_task_list=[extract_future_gcm_task,
                                     extract_historical_gcm_task]
            )

            reduce_gcm_task = graph.add_task(
                func=knn.reduce_netcdf,
                kwargs={
                    'source_file_list': [
                        gcm_historical_extract_path, gcm_future_extract_path],
                    'aoi_netcdf_path': aoi_mask_gcm_path,
                    'target_filepath': gcm_netcdf_path,
                },
                task_name='Reduce GCM to average value within AOI.',
                target_path_list=[gcm_netcdf_path],
                dependent_task_list=[
                    rasterize_aoi_gcm_task,
                    extract_historical_gcm_task,
                    extract_future_gcm_task]
            )

            bootstrap_dates_task = graph.add_task(
                func=knn.bootstrap_dates_precip,
                kwargs={
                    'observed_data_path': mswep_netcdf_path,
                    'prediction_dates': prediction_dates,
                    'reference_period_dates': reference_period_dates,
                    'gcm_netcdf_path': gcm_netcdf_path,
                    'lower_precip_threshold': args['lower_precip_threshold'],
                    'upper_precip_percentile': args['upper_precip_percentile'],
                    'target_csv_path': target_csv_path
                },
                task_name='Bootstrap dates for precipitation',
                target_path_list=[target_csv_path],
                dependent_task_list=[reduce_gcm_task, reduce_mswep_task]
            )

            # TODO: is it problematic for these task objects created in for-loop
            # to overwrite each other?
            extreme_precip_threshold = extreme_values_task.get()
            downscale_precip_task = graph.add_task(
                func=knn.downscale_precip,
                kwargs={
                    'bootstrapped_dates_path': target_csv_path,
                    'gridded_observed_precip': mswep_extract_path,
                    'aoi_mask_path': aoi_mask_mswep_path,
                    'target_netcdf_path': target_netcdf_path,
                    'extreme_value_samples_path': target_extreme_values_path,
                    'extreme_precip_threshold': extreme_precip_threshold
                },
                task_name='Downscale Precipitation',
                target_path_list=[target_netcdf_path],
                dependent_task_list=[bootstrap_dates_task, extreme_values_task]
            )

            target_pdf_path = file_registry[
                ('downscaled_precip_[MODEL]_[EXPERIMENT].pdf',
                 gcm_model, gcm_experiment)]

            report_task = graph.add_task(
                func=plot.plot,
                kwargs={
                    'dates_filepath': target_csv_path,
                    'precip_filepath': target_netcdf_path,
                    'observed_mean_precip_filepath': mswep_netcdf_path,
                    'observed_precip_filepath': mswep_extract_path,
                    'aoi_netcdf_path': aoi_mask_mswep_path,
                    'reference_period_dates': reference_period_dates,
                    'hindcast': False,
                    'target_filename': target_pdf_path
                },
                task_name='Report',
                target_path_list=[target_pdf_path],
                dependent_task_list=[downscale_precip_task]
            )

    graph.close()
    graph.join()
    return file_registry.registry


@validation.invest_validator
def validate(args, limit_to=None):
    return validation.validate(args, MODEL_SPEC)

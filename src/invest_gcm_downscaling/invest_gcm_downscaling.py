"""
This work is an adaptation of the GCMClimTool Library
by Angarita H., Yates D., Depsky N. 2014-2021
"""

import logging
from pprint import pformat
import pandas
import warnings

from knn import knn

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

DATE_EXPR = r"^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"

MODEL_LIST = [
    'CanESM5',  # Canadian Earth System Model version 5
    'CESM2',  # NCAR (National Center for Atmospheric Research) Community Earth System Model 2
    'CESM2-WACCM',  # NCAR Community Earth System Model 2 - Whole Atmosphere Community Climate Model
    'CMCC-CM2-HR4',  # CMCC (Euro-Mediterranean Center on Climate Change) Climate Model 2 - High Resolution 4
    'CMCC-CM2-SR5',  # CMCC Climate Model 2 - Standard Resolution
    'CMCC-ESM2',  # CMCC Earth System Model 2
    'FGOALS-g3',  # Flexible Global Ocean-Atmosphere-Land System Model Grid-Point Version 3
    'GFDL-EMS4',  # NOAA Geophyscial Fluid Dynamics Laboratory Earth System Model 4
    'MIROC6',  # Model for Interdisciplinary Research on Climate 6 (Japan)
    'MPI-ESM1-2-LR',  # Max Planck Institute Earth System Model 2 - Low Res
    # 'CESM-FV2',  # never created zarrs; https://github.com/natcap/gcm-downscaling/issues/19
    # 'IPSL-CM6A-LR', # unreadable by xarray; https://github.com/h5netcdf/h5netcdf/issues/94
    # 'MPI-ESM1-2-HR'  # has a bad file with 0 bytes
]


MODEL_SPEC = spec.ModelSpec(
    model_id='gcm_downscaling',
    model_title=gettext('GCM Downscaling'),
    userguide='',  # change to README later
    input_field_order=[
        ['workspace_dir', 'aoi_path'],
        ['reference_period_start_date', 'reference_period_end_date'],
        ['prediction_start_date', 'prediction_end_date'],
        ['hindcast'],
        ['gcm_model'],
        ['upper_precip_percentile', 'lower_precip_threshold'],
        ['observed_dataset_path']],
    inputs=[
        spec.DirectoryInput(
            id="workspace_dir",
            name=gettext("workspace"),
            about=gettext(
                "The folder where all the model's output files will be written. If "
                "this folder does not exist, it will be created. If data already "
                "exists in the folder, it will be overwritten."),
            contents={},
            must_exist=False,
            permissions="rwx"
        ),
        spec.NumberInput(
            id="n_workers",
            name=gettext("taskgraph n_workers parameter"),
            about=gettext(
                "The n_workers parameter to provide to taskgraph. "
                "-1 will cause all jobs to run synchronously. "
                "0 will run all jobs in the same process, but scheduling will take "
                "place asynchronously. Any other positive integer will cause that "
                "many processes to be spawned to execute tasks."),
            units=None,
            required=False,
            expression="value >= -1",
            hidden=True
        ),
        spec.VectorInput(
            id='aoi_path',
            name='Area of Interest',
            about=gettext('Area of interest'),
            required=True,
            fields={},
            geometry_types={'POLYGON', 'MULTIPOLYGON'}
            # 'projected'=True,
        ),
        spec.StringInput(
            id='reference_period_start_date',
            name='Reference Period Start Date',
            about=gettext('First day in the reference period, which is '
                        'used to calculate climate "normals". The reference '
                        'period should typically span about 30 years or more. '
                        'If ``observed_dataset`` is not input, reference period '
                        'must extend past 1979, and should extend at least 30 years beyond 1979.'
                        'Format: "YYYY-MM-DD"'),
            required=True,
            regexp=DATE_EXPR
        ),
        spec.StringInput(
            id='reference_period_end_date',
            name='Reference Period End Date',
            about=gettext('Last day in the reference period, which is '
                        'used to calculate climate "normals". The reference '
                        'period should typically span about 30 years or more. '
                        'If ``observed_dataset`` is not input, reference period '
                        'must extend past 1979, and should extend at least 30 years beyond 1979.'
                        'Format: "YYYY-MM-DD"'),
            required=True,
            regexp=DATE_EXPR
        ),
        spec.StringInput(
            id='prediction_start_date',
            name='Prediction Start Date',
            about=gettext("First day in the simulation period, in format 'YYYY-MM-DD'"),
            required='not hindcast',
            regexp=DATE_EXPR
        ),
        spec.StringInput(
            id='prediction_end_date',
            name='Prediction End Date',
            about=gettext("Last day in the simulation period, in format 'YYYY-MM-DD'"),
            required='not hindcast',
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
            options=MODEL_LIST,
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
            required=True
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
            id='downscaled_precip_[model]_[experiment].nc',

            about=gettext(
                'Gridded NetCDF file containing the downscaled daily '
                'precipitation time series for the specified climate '
                'model and experiment scenario.')
        ),
        spec.FileOutput(
            id='downscaled_precip_[model]_[experiment].pdf',
            about=gettext(
                'Report with graphs and visualizations of downscaled '
                'precipitation data for specified model and experiment')
        ),
        spec.SingleBandRasterOutput(
            id='downscaled_precip_hindcast.nc',
            about=gettext(
                'Gridded NetCDF file with downscaled historical '
                'precipitation data (hindcast), serving as a baseline for '
                'model validation.')
        ),
        spec.FileOutput(
            id='downscaled_precip_hindcast.pdf',
            about=gettext(
                'Report with graphs and visualizations of downscaled '
                'hindcast precipitation data.')
        )
        # spec.DirectoryOuptut(
        #     contents=[
        #     ]
        # )
    ]
)
    # 'aoi_mask_[model].nc': {
    #     'about': 'Area of Interest (AOI) mask'
    # },
    # 'bootstrapped_dates_precip_[model_experiment | hindcast].csv': {
    #     'about': 'Bootstrapped dates and associated precipitation values used '
    #                 'in the downscaling process.',
    #     'columns': {
    #         'historic_date': {
    #             'about': 'Date from the historical record used in bootstrapping.',
    #             'type': 'freestyle_string'  # date
    #         },
    #         'historic_precip': {
    #             'about': 'Historic precipitation',
    #             'type': 'number',
    #             'unit': u.millimeter
    #         },
    #         'wet_state': {
    #             'about': 'Dry/wet/very wet state classification for the historic date.',
    #             'type': 'integer'
    #         },
    #         'next_wet_state': {
    #             'about': 'Predicted dry/wet/very wet state for the subsequent time step.',
    #             'type': 'integer'
    #         },
    #         'next_historic_date': {
    #             'about': 'Next date in the bootstrapped sequence.',
    #             'type': 'freestyle_string'  # date
    #         }
    #     }
    # },
    # 'extracted_[model]_[experiment | hindcast].nc': {
    #     'about': 'NetCDF file containing precipitation data extracted from the '
    #                 'specified model and experiment (or hindcast), prior to downscaling.'
    # },
    # 'extracted_mswep.nc': {
    #     'about': 'NetCDF file with precipitation data extracted from the MSWEP '
    #                 'dataset, used as observational reference.'
    # },
    # 'mswep_mean.nc': {
    #     'about': 'NetCDF file representing the mean precipitation from the '
    #                 'MSWEP dataset over the analysis period.'
    # },
    # 'pr_day_[model]_[experiment]_mean.nc': {
    #     'about': 'NetCDF file containing the daily mean precipitation values '
    #                 'for the specified model and experiment.'
    # },
    # 'synthesized_extreme_precip_[model]_[experiment].csv': {
    #     'about': 'CSV file summarizing synthesized extreme precipitation events '
    #                 'for the specified model and experiment.',
    #     'columns': {
    #         'historic_sample': {
    #             'about': 'Precipitation value from the historical sample',
    #             'type': 'number',
    #             'units': u.millimeters
    #         },
    #         'forecast_sample': {
    #             'about': 'Projected precipitation value from the forecast sample',
    #             'type': 'number',
    #             'units': u.millimeters
    #         }
    #     }
    # }


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
            range of the observed dataset (``MSWEP_DATE_RANGE``).
        args['prediction_start_date'] (string, optional):
            ('YYYY-MM-DD') first day in the simulation period.
            Required if `hindcast=False`.
        args['prediction_end_date'] (string, optional):
            ('YYYY-MM-DD') last day in the simulation period.
            Required if `hindcast=False`.
        args['gcm_model'] (sequence, optional): a sequence of strings
            representing CMIP6 model codes. Each model will be used to generate
            a single downscaled product for each experiment in `gcm_experiment_list`.
            Available models are stored in ``GCM_MODEL_LIST``.
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
            processes, which can be useful if `gcm_experiement_list` contain
            multiple items.
    """
    LOGGER.info(pformat(args))

    ref_end = pandas.to_datetime(args['reference_period_end_date'])
    ref_start = pandas.to_datetime(args['reference_period_start_date'])

    # check that end dates are after start dates
    if ref_end <= ref_start:
        raise ValueError('Reference end date must be after reference start date.')

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

    model_args = {
        'aoi_path': args['aoi_path'],
        'workspace_dir': args['workspace_dir'],
        'reference_period_dates': (args['reference_period_start_date'],
                                   args['reference_period_end_date']),
        'prediction_dates': (args['prediction_start_date'],
                             args['prediction_end_date']),
        'hindcast': args['hindcast'],
        'gcm_experiment_list': knn.GCM_EXPERIMENT_LIST,
        'gcm_model_list': [args['gcm_model']],
        'upper_precip_percentile': float(args['upper_precip_percentile']),
        'lower_precip_threshold': float(args['lower_precip_threshold']),
        'observed_dataset_path': args['observed_dataset_path'] or None,
        'n_workers': args['n_workers'],
    }

    knn.execute(model_args)



@validation.invest_validator
def validate(args):
    pass
    # return validation.validate(args, MODEL_SPEC) #['args'])
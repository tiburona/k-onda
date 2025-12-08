from collections import defaultdict
from copy import deepcopy
import importlib
import json
import numpy as np
import pickle
import os
import re
from pathlib import PosixPath
import xarray as xr
import pint
import pint_xarray

from k_onda.utils import  safe_make_dir
from k_onda.math import Filter
    

class Base:
    
    _experiment = None
    _calc_opts = {}
    _io_opts = {}
    _env_config = {}
    _cache = defaultdict(dict)
    _criteria = {}
    _selected_conditions = {}
    _selected_period_type = ''
    _selected_period_types = []
    _selected_period_group = []
    _selected_neuron_type = ''
    _selected_brain_region = ''
    _selected_frequency_band = ''
    _selected_region_set = []

    _shared_filters = {}
    original_periods = None
    selectable_variables = [
        'period_type', 
        'period_types',
        'period_conditions', 
        'period_group',
        'neuron_type', 
        'conditions', 
        'brain_region', 
        'region_set',
        'frequency_band'
        ]
    
    def __init__(self, **_):
        super().__init__()
    
    @property
    def experiment(self):
        return Base._experiment

    @experiment.setter
    def experiment(self, value):
        Base._experiment = value

    @property
    def calc_opts(self):
        return Base._calc_opts  
    
    @calc_opts.setter
    def calc_opts(self, value):
        Base._calc_opts = value
        self.set_criteria_from_calc_opts()
        Base._cache = defaultdict(dict)

    @property
    def io_opts(self):
        return Base._io_opts
    
    @io_opts.setter
    def io_opts(self, value):
        Base._io_opts = value

    @property
    def ureg(self):
        return self.experiment._ureg
    
    def quantity(
        self,
        value,
        units: str,
        name: str | None = None,
        dims=None,
        coords=None,
    ):
        """
        Wrap `value` in an xarray.DataArray with units, then quantify with pint.

        - If `value` is a DataArray, preserve its dims/coords (ignore dims/coords args).
        - If `value` is array-like, use provided dims/coords and check they match.
        """
        if isinstance(value, xr.DataArray):
            da = value.copy()
            if name is not None:
                da = da.rename(name)
            da.attrs["units"] = units
        else:
            data = np.asarray(value)

            # normalize dims
            if dims is None:
                # scalar -> 0-D, array -> need explicit dims
                if data.ndim == 0:
                    dims = ()
                else:
                    raise ValueError(
                        f"dims must be provided for non-scalar data (got shape {data.shape})"
                    )
            elif isinstance(dims, str):
                dims = (dims,)

            if data.ndim != len(dims):
                raise ValueError(
                    f"different number of dimensions on data and dims: "
                    f"{data.ndim} vs {len(dims)} (dims={dims}, shape={data.shape})"
                )

            da = xr.DataArray(
                data,
                name=name,
                dims=dims,
                coords=coords,
                attrs={"units": units},
            )

        return da.pint.quantify(unit_registry=self.ureg)
    
    def to_int(self, q, unit=None):
        """
        Convert a pint-xarray quantity `q` to int(s) in the given unit.

        - Scalar -> Python int
        - Array  -> numpy.ndarray[int]
        """
        return self.to_numerical_type(q, unit, dtype="int")
 
    def to_float(self, q, unit=None, decimals=None):
        """
        Convert a pint-xarray quantity `q` to float(s) in the given unit.

        - If unit is provided, convert to that unit first.
        - Scalar -> Python float
        - Array  -> numpy.ndarray[float]
        """
        return self.to_numerical_type(q, unit, dtype="float", decimals=decimals)
       
    def to_numerical_type(self, q, unit=None, dtype="float", decimals=None):
        """
        Convert a pint-xarray quantity or xarray.DataArray `q` to floats or ints
        in the given unit (if pint-quantified).

        - q can be:
            * pint-xarray DataArray
            * plain xarray.DataArray
            * numpy array / scalar
        - unit:
            * only used if `q` is pint-quantified; otherwise must be None
        - dtype = "float" or "int"
            * Scalar -> Python scalar
            * Array  -> numpy.ndarray
        - decimals:
            * if not None and dtype=="float", round to this many decimals
        """
        # Is this a pint-quantified xarray object?
        is_pint = hasattr(q, "pint") and hasattr(q.pint, "units")

        # Optional unit conversion (only valid for pint objects)
        if unit is not None:
            if is_pint:
                q = q.pint.to(unit)
            else:
                raise TypeError("`unit` is only valid when `q` is a pint-quantified DataArray.")

        # Extract raw magnitude / data
        if is_pint:
            mag = q.pint.magnitude           # numpy scalar or ndarray
        elif isinstance(q, xr.DataArray):
            mag = q.data                     # xarray -> numpy
        else:
            mag = q                          # already numpy or scalar

        mag = np.asarray(mag)
        size = mag.size

        if dtype == "float":
            if decimals is not None:
                mag = np.round(mag, decimals=decimals)

            if size == 1:
                return float(mag)
            else:
                return mag.astype(float)

        elif dtype == "int":
            if size == 1:
                return int(np.rint(float(mag)))
            else:
                return np.rint(mag).astype(int)

        else:
            raise ValueError(f"Unknown data type: {dtype!r}")
        
    def standardize_time(self, q, units="second", decimals=8):
        """
        Round a pint-xarray time quantity to a fixed number of decimals,
        preserving units, dims, and coords.
        """
        # ensure we’re working in the desired unit
        q2 = q.pint.to(units)

        # round magnitudes
        mag = np.round(q2.pint.magnitude, decimals=decimals)

        # rebuild a DataArray with the same structure
        da = xr.DataArray(
            mag,
            dims=q2.dims,
            coords=q2.coords,
            name=q2.name,
            attrs={"units": units},  # <-- IMPORTANT: "units", not "unit"
        )

        # re-quantify with pint-xarray
        return da.pint.quantify(unit_registry=self.ureg)
    
    def rebind_to_ureg(self, da: xr.DataArray) -> xr.DataArray:
        """
        Ensure `da` is quantified with this object's unit registry.

        - If it's already pint-quantified with some registry, dequantify -> quantify.
        - If it's plain xarray with "units"/"unit" attrs, just quantify.
        """
        # try to drop any existing pint wrapping
        try:
            da = da.pint.dequantify()
        except Exception:
            # either not quantified or pint_xarray not attached; that's fine
            pass

        # now quantify with *our* registry using attrs["units"] / ["unit"]
        return da.pint.quantify(unit_registry=self.ureg)

    @property
    def env_config(self):
        return Base._env_config
    
    @env_config.setter
    def env_config(self, value):
        Base._env_config = value

    @property
    def cache(self):
        return Base._cache
    
    def clear_cache(self):
        Base._cache = defaultdict(dict)

    @property
    def criteria(self):
        return Base._criteria
    
    @criteria.setter
    def criteria(self, criteria):
        Base._criteria = criteria

    def set_criteria_from_calc_opts(self):
        self.criteria = defaultdict(lambda: defaultdict(tuple))
        criteria = self.calc_opts.get('criteria', {})
        if not criteria:
            return
        if isinstance(criteria, list):
            for criterion in criteria:
                self.add_to_criteria(self.parse_natural_language_criteria(criterion))
        else:
            for object_type in criteria:
                object_criteria = self.calc_opts['criteria'][object_type]
                for property in object_criteria:
                    self.criteria[object_type][property] = object_criteria[property]   
            if self.calc_opts.get('validate_events'):
                self.criteria['event']['is_valid'] = ('==', True)

    def add_to_criteria(self, obj_name, attr, operator, target_val):
       
        if attr in ['conditions', 'period_types'] and attr in self.criteria[obj_name]:
            self.criteria[obj_name][attr][1].update(target_val)

        else:
            self.criteria[obj_name][attr] = (operator, target_val)
              
    def del_all_criteria(self):
        self.criteria = defaultdict(lambda: defaultdict(tuple))

    def del_from_criteria(self, obj_name, attr):
        del self.criteria[obj_name][attr]

    @property
    def shared_filters(self):
        return Base._shared_filters
    
    @staticmethod
    def filter_key(cfg: dict):
        # Only params that change coefficients go in the key
        return (
            float(cfg["fs"]), float(cfg["low"]), float(cfg["high"]),
            cfg.get("method", "fir_hamming"),
            cfg.get("numtaps"),
            cfg.get("iir_order"),
            cfg.get("iir_ripple_db"),
            cfg.get("kaiser_atten_db"),
            cfg.get("kaiser_tw_hz"),
            cfg.get("notch_Q"),
            cfg.get("_version", "v1"),
        )
    
    def get_or_create_filter(self, cfg: dict) -> Filter:
        k = self.filter_key(cfg)
        flt = self.shared_filters.get(k)
        if flt is None:
            flt = Filter(cfg)          # designs once
            self.shared_filters[k] = flt
        return flt

    @property
    def kind_of_data(self):
        return self.calc_opts.get('kind_of_data')

    @property
    def calc_type(self):
        return self.calc_opts['calc_type']

    @calc_type.setter
    def calc_type(self, calc_type):
        self.calc_opts['calc_type'] = calc_type

    @property
    def selected_conditions(self):
        return self._selected_conditions
    
    @selected_conditions.setter
    def selected_conditions(self, conditions):
        Base._selected_conditions = conditions
        self.add_to_criteria('animal', 'conditions', 'partial_dict_match', conditions)

    @property
    def selected_neuron_type(self):
        return Base._selected_neuron_type

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        Base._selected_neuron_type = neuron_type

    @property
    def selected_period_type(self):
        return Base._selected_period_type
        
    @selected_period_type.setter
    def selected_period_type(self, period_type):
        Base._selected_period_types = []
        Base._selected_period_type = period_type

    @property
    def selected_period_types(self):
        return Base._selected_period_types
        
    @selected_period_types.setter
    def selected_period_types(self, period_types):
        Base._selected_period_type = ''
        Base._selected_period_types = period_types

    @property
    def selected_period_conditions(self):
        return Base._selected_period_conditions
        
    @selected_period_types.setter
    def selected_period_conditions(self, period_conditions):
        Base._selected_period_conditions = period_conditions

    @property
    def selected_period_group(self):
        return Base._selected_period_group
    
    @selected_period_group.setter
    def selected_period_group(self, period_group):
        Base._selected_period_group = period_group
    
    @property
    def selected_frequency_band(self):
        return self.calc_opts['frequency_band']

    @selected_frequency_band.setter
    def selected_frequency_band(self, frequency_band):
        self.calc_opts['frequency_band'] = frequency_band

    @property
    def selected_brain_region(self):
        return self.calc_opts.get('brain_region')
    
    @selected_brain_region.setter
    def selected_brain_region(self, brain_region):
        self.calc_opts['brain_region'] = brain_region

    @property
    def selected_region_set(self):
        return self.calc_opts.get('region_set')

    @selected_region_set.setter
    def selected_region_set(self, region_set):
        self.calc_opts['region_set'] = region_set

    @property
    def selected_brain_regions(self):
        if self.selected_region_set is not None:
            return self.selected_region_set.split('_')
        else:
            return [self.selected_brain_region]

    @property
    def frequency_band_definition(self):
        return self.calc_opts.get('frequency_band_definition') or \
            self.experiment.exp_info['frequency_bands'] or {}
           
    @property
    def freq_range(self):
        if isinstance(self.selected_frequency_band, str):
            freq_range = self.frequency_band_definition[self.selected_frequency_band]
        else:
            freq_range = self.selected_frequency_band
        
        return self.quantity(
            freq_range,
            units='Hz',
            dims=('edge',),
            coords={'edge': ['low', 'high']},
            name='frequency_range'
        )
    
    def get_data_sources(self, data_object_type=None, identifiers=None, identifier=None):
        if data_object_type is None:
            data_object_type = self.calc_opts['base']
            if data_object_type in ['period', 'event']:
                data_object_type = f"{self.kind_of_data}_{data_object_type}"
        data_sources = getattr(self.experiment, f"all_{data_object_type}s")
        if identifier and 'all' in identifier:
            return data_sources
        if identifiers:
            return [source for source in data_sources if source.unique_id in identifiers]
        if identifier:
            return [source for source in data_sources if source.unique_id == identifier][0]

    @property
    def post(self):
        if self.calc_opts.get('base') == 'period':
            return self.post_period
        
    @property
    def pre_event(self):
        return self.get_pre_post(0, 'event')
    
    @property
    def post_event(self):
        return self.get_pre_post(1, 'event')

    @property
    def pre_period(self):
        return self.get_pre_post(0, 'period')
    
    @property
    def post_period(self):
        return self.get_pre_post(1, 'period')
    
    def get_pre_post(self, time, obj_type):
        pt = getattr(self, 'period_type', None)
        if pt is None and hasattr(self, 'parent'):
            pt = getattr(self.parent, 'period_type', None)
        if pt is None:
            pt = self.selected_period_type

        pre_post = (
            self.calc_opts
            .get('periods', {})
            .get(pt, {})
            .get(f'{obj_type}_pre_post', (0, 0))[time]
        )

        return self.quantity(
            pre_post,
            units="second",
            name=f"{obj_type}_pre_post",
        )
        
    @property
    def bin_size(self):
        bin_size = self.calc_opts.get('bin_size', .01)
        return self.quantity(bin_size, units='second', name='bin_size')
    
    def load(self, path_id, calc_name, other_identifiers=None):
        store = self.calc_opts.get('store', 'pkl')
        data_path = self.construct_path(path_id)
        store_dir = os.path.join(data_path, f"{calc_name}_{store}s")
        if other_identifiers is None:
            other_identifiers = []
        for p in [data_path, store_dir]:
            if not os.path.exists(p):
                safe_make_dir(p)
        store_path = os.path.join(store_dir, '_'.join(other_identifiers) + f".{store}")
        if os.path.exists(store_path) and not self.calc_opts.get('force_recalc'):
            with open(store_path, 'rb') as f:
                if store == 'pkl':
                    return_val = pickle.load(f)
                else:
                    return_val = json.load(f)
                return True, return_val, store_path
        else:
            return False, None, store_path

    def save(self, result, store_path):
        store = self.calc_opts.get('store', 'pkl')
        mode = 'wb' if store == 'pkl' else 'w'
        with open(store_path, mode) as f:
            if store == 'pkl':
                return pickle.dump(result, f)
            else:
                result_str = json.dumps([arr.tolist() for arr in result])
                f.write(result_str)

    def load_user_module(file_path):
        # Extract a module name from the file path (e.g., "user_plugin")
        module_name = os.path.splitext(os.path.basename(file_path))[0]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            raise ImportError(f"Could not load module from {file_path}")
        
    def construct_path(self, constructor_id):
        for key, val in self.io_opts.get('paths', {}).items():
            if constructor_id == key:
                constructor = deepcopy(val)
                return self.fill_fields(constructor)
        constructor = deepcopy(self.experiment.exp_info['paths'][constructor_id])
        return self.fill_fields(constructor)
    
    @staticmethod
    def preprocess_constructor(constructor, placeholder="__"):
        # Replace dots in keys
        processed_constructor = {
            key.replace(".", placeholder): value for key, value in constructor.items()
        }
        # Replace dots in the template
        processed_constructor['template'] = processed_constructor['template'].replace(".", placeholder)
        return processed_constructor
    
    def fill_fields(self, constructor, obj=None, **kwargs):
        if not constructor:
            return
        if not obj:
            obj = self

        if isinstance(constructor, (str, PosixPath)):
            constructor = str(constructor)
            if '{' not in constructor:
                return constructor
            else:
                constructor = {
                    'template': constructor, 
                    'fields': re.findall(r'\{(.*?)\}', constructor)
                }

        new_fields = {}
        lambda_counter = 0

        for field in constructor['fields']:
            if 'lambda' in field:
                # Create a valid placeholder key.
                key = f'_lambda{lambda_counter}'
                lambda_counter += 1
                # Replace the lambda expression placeholder in the template.
                constructor['template'] = constructor['template'].replace(f'{{{field}}}', f'{{{key}}}')
                new_fields[key] = eval(field)(obj)
            elif field in kwargs:
                new_fields[field] = kwargs[field]
            elif field in obj.selectable_variables:
                new_fields[field] = getattr(obj, field, getattr(self, f'selected_{field}'))
            elif '.' in field:
                attrs = field.split('.').reverse()
                new_attr = obj
                while len(attrs):
                    new_attr = getattr(new_attr, attrs.pop())
                new_fields[field] = new_attr
            elif '|' in field:
                field_type, field_key = field.split('|')
                new_fields[field] = getattr(obj, f'selected_{field_type}')[field_key]
            else:
                try:
                    new_fields[field] = getattr(obj, field)
                except AttributeError as e:
                    ds_dict = kwargs['data_source_dict']
                    data_sources = self.get_data_sources(data_object_type=ds_dict['data_source'], 
                                                         identifiers=ds_dict['members'])
                    new_fields[field] = '_'.join([getattr(ds, field) for ds in data_sources])

        constructor.update(new_fields)
        return constructor['template'].format(**constructor)
    

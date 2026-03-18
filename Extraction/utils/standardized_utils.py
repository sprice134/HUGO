# Created by Kyle Miller on March 18th, 2026
# Released under the Apache 2.0 License

import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import argparse

def parse_nonnumerical_values(row, target_key):
    """Parse a non-numerical value.
    Example Usage: parsed_vals, parsed_units, parsed_uncertainties, err_messages_value, err_messages_unit, pass_thru_msgs = zip(*df.apply(parse_nonnumerical_values, args=('resultsValues.Deposit_Tensile_Test_Strain_Rate',), axis=1))
    Args:
        row (pd.Series): Row containing the value and unit to parse.
        target_key (str): Target key to parse.
    """
    pass


def parse_values_and_units(row, target_key):
    """
    Parse a numerical value and unit.
    Returns:
        parsed_value (float|str): float for parsed values; '' if input was blank
        parsed_unit (str): normalized unit or original; '' if input was blank
        parsed_uncertainty (float|str): float when parsed; '' if input was blank
        err_msg_value (str): '' for blanks; error text for true parsing errors
        err_msg_unit (str): '' for blanks; error text for unit conversion errors
        pass_thru_msg (str): diagnostic tag (not persisted to output)
    """
    if len(target_key) == 0:
        raise ValueError('target_key is empty')

    raw_value = row.get(f'{target_key}_Value', '')
    raw_unit  = row.get(f'{target_key}_Units', '')
    measurement_type = target_key
    pass_thru_msg = row.get('filename', '')
    additional_info_dict = {}

    # NEW: treat blanks as blanks (do not mark invalid/do not error)
    is_blank = (
        raw_value is None
        or (isinstance(raw_value, str) and raw_value.strip() in ('', 'nan', 'NaN'))
        or (isinstance(raw_value, float) and np.isnan(raw_value))
    )
    if is_blank:
        # keep value/unit/uncertainty blank and error messages empty
        out_val = ''
        out_unit = raw_unit if isinstance(raw_unit, str) else ''
        out_unc = ''
        return out_val, out_unit, out_unc, '', '', pass_thru_msg

    # (everything below here is your existing parsing logic for real, nonblank values)
    # Try parsing as a lone number
    if isinstance(raw_value, (int, float)):
        parsed_value = float(raw_value)
        parsed_uncertainty = np.nan
        err_msg_value= ''
        if np.isnan(parsed_value):
            # For safety: double-check; but blanks already returned above
            return '', str(raw_unit) if isinstance(raw_unit, str) else '', '', '', '', pass_thru_msg

    elif isinstance(raw_value, str):
        # Special case strings
        if 'temperature' in measurement_type.lower():
            if 'room' in raw_value.lower() or 'ambient' in raw_value.lower() or 'rt' in raw_value.lower():
                return 20, '°C', np.nan, '', '', pass_thru_msg
        if 'angle' in measurement_type.lower():
            if raw_value.lower() in ['perpendicular']:
                return 90, 'deg', np.nan, '', '', pass_thru_msg


    ### Define regexes
    value_w_uncert_re = re.compile(r'^([+-]?\d+(?:\.\d+)?)\s*(?:\(\s*(?:\+/-|±)\s*\)|(?:\+/-|±))\s*(\d+(?:\.\d+)?)')
    value_lte_re = re.compile(r'(?:<|<=|≤)\s*([+-]?\d+(?:\.\d+)?)')
    value_gte_re = re.compile(r'(?:>|>=|≥)\s*([+-]?\d+(?:\.\d+)?)')
    numrange_re = re.compile(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)')
    num_re = re.compile(r'^([0-9.]+)')    

    ### Try extracting a range of values -- if it is a range, we will take the mean and uncertainty as the mean of the range and half the range, respectively
    if numrange_re.match(str(raw_value).strip()):
        match = numrange_re.match(str(raw_value).strip())
        parsed_value = (float(match.group(1)) + float(match.group(2))) / 2
        parsed_uncertainty = (float(match.group(2)) - float(match.group(1))) / 2
        err_msg_value = ''
        
    ### If it's not a range, we can strip '≈' and '~'
    else:
        ### Strip meaningless characters
        chars_to_strip = ['≈','~']
        proc_value = str(raw_value)
        # print(f'{raw_value} -> {proc_value}')
        for char in chars_to_strip:
            proc_value = proc_value.replace(char, '')

        ### Parse value and uncertainty
        if value_w_uncert_re.match(str(proc_value).strip()):
            match = value_w_uncert_re.match(str(proc_value).strip())
            parsed_value = float(match.group(1))
            parsed_uncertainty = float(match.group(2))
            err_msg_value= ''
        
        ### Try extracting just a single value
        elif num_re.match(str(proc_value).strip()):
            match = num_re.match(str(proc_value).strip())
            parsed_value = float(match.group(1))
            parsed_uncertainty = np.nan
            err_msg_value= ''

        ### Try extracting a value less than or equal to a value
        elif value_lte_re.match(str(proc_value).strip()):
            match = value_lte_re.match(str(proc_value).strip())
            parsed_value = float(match.group(1))
            parsed_uncertainty = np.nan
            err_msg_value= 'less than'
        
        ### Try extracting a value greater than or equal to a value
        elif value_gte_re.match(str(proc_value).strip()):
            match = value_gte_re.match(str(proc_value).strip())
            parsed_value = float(match.group(1))
            parsed_uncertainty = np.nan
            err_msg_value= 'greater than'

        else:
            parsed_value = np.nan
            parsed_uncertainty = np.nan
            err_msg_value = f'Unknown value: {measurement_type}, {raw_value} {raw_unit}, {pass_thru_msg}'
            print(err_msg_value)
            return np.nan, 'invalid', np.nan, err_msg_value, 'invalid value', pass_thru_msg

    ### Convert unit
    additional_info_dict = {}
    if measurement_type == 'experimentalProperties.Deposit_Tensile_Test_Strain_Rate':
        additional_info_dict = {
            'experimentalProperties.Tensile_Specimen_Gauge_Length_Value': row['experimentalProperties.Tensile_Specimen_Gauge_Length_Value'],
            'experimentalProperties.Tensile_Specimen_Gauge_Length_Units': row['experimentalProperties.Tensile_Specimen_Gauge_Length_Units']
        }
    parsed_value, parsed_unit, parsed_uncertainty, err_msg_unit = convert_unit(parsed_value, raw_unit, uncertainty=parsed_uncertainty, measurement_type=measurement_type, pass_thru_msg=pass_thru_msg, additional_info_dict=additional_info_dict)
    
    ### Custom interdependent unit conversions
    ### Hardness test load units
    if 'deposit_microhardness' in measurement_type.lower():
        # Check for parsed test load value
        if 'load' in parsed_unit:
            load_value = float(parsed_unit.split('load_')[1].split('_units_')[0])
            load_units = parsed_unit.split('units_')[1]        
            # Extract test load value
            test_load_value_key = measurement_type.replace('resultsValues.', 'experimentalProperties.').replace('hardness', 'hardness_Test_Load_Value')
            test_load_units_key = measurement_type.replace('resultsValues.', 'experimentalProperties.').replace('hardness', 'hardness_Test_Load_Units')
            # If no existing test load value, use the parsed value
            # print(f'Examining value: {row[test_load_value_key]}, units: {row[test_load_units_key]}')
            # print(f'Parsed value:  {parsed_unit} --> {load_value} {load_units}')
            if row[test_load_value_key] == '':
                print(f'Using parsed test load value: {load_value} {load_units}')
                row[test_load_value_key] = load_value
                row[test_load_units_key] = load_units
            elif np.isnan(float(row[test_load_value_key])):
                print(f'Using parsed test load value: {load_value} {load_units}')
                row[test_load_value_key] = load_value
                row[test_load_units_key] = load_units

    if err_msg_value == 'missing value':
        print(f'MISSING VALUE: {raw_value} {raw_unit} -> {parsed_value} {parsed_unit} +/- {parsed_uncertainty} -- {err_msg_value} -- {err_msg_unit} --  {pass_thru_msg}')
    # else:
    #     print(f'{parsed_value} {parsed_unit} {parsed_uncertainty} {err_msg_value} {err_msg_unit} {pass_thru_msg}')
    return parsed_value, parsed_unit, parsed_uncertainty, err_msg_value, err_msg_unit, pass_thru_msg


def convert_unit(val, unit, uncertainty=np.nan, measurement_type='unknown', 
                 pass_thru_msg='', additional_info_dict={}):
    """Convert value to standard units
    Args:
        val (float): Value to convert.
        unit (str): Unit of the value.
        uncertainty (float): Uncertainty of the value.
        measurement_type (str): Measurement type.
        pass_thru_msg (str): Message to pass through (for diagnostic purposes)
        additional_info_dict (dict): Additional information about the value (for conversion purposes)
    Returns:
        float: Value in standard units.
        str: Unit in standard units.
        float: Uncertainty in standard units.
        str: Error message describing the conversion issue
    """

    ### Check for invalid type (not string)
    if not isinstance(unit, str):
        return val, 'invalid', uncertainty, f'non-string unit: {unit}'
    
    ### Uniformize and clean spaces from the unit string
    case_sensitive_u = (unit or '').strip()
    u = (unit or '').strip().lower()
    
    ### Detect known invalid units
    if unit in ['', ' ', 'nan']:
        return val, 'invalid', uncertainty, 'empty'
    
    ### Porosity
    if 'porosity' in measurement_type.lower():
        porosity_unknown = {'‰': 0.1,'%': 1, 'pct': 1, 'percent': 1, '0.01%': 0.01}
        if u in porosity_unknown:
            return val * porosity_unknown[u], '%', uncertainty * porosity_unknown[u], ''

        porosity_abs_area = {'unit/cm2': 1}   
        porosity_area = {'% area' : 1, 'area %' : 1, '%area' : 1, 'area%' : 1} 
        if u in porosity_abs_area or u in porosity_area:
            # return val * porosity_abs_area[u], '1/cm^2', uncertainty * porosity_abs_area[u]
            return val, 'invalid', uncertainty, f'incompatible unit: {case_sensitive_u}'

        porosity_volume = {'vol%' : 1, 'vol.%' : 1, '%vol' : 1, 'vol. %' : 1, 'vol %' : 1, '% vol' : 1}
        if u in porosity_volume:
            return val * porosity_volume[u], '%', uncertainty * porosity_volume[u], ''
    
    ### Grain/particle size
    if 'size' in measurement_type.lower() and ('grain' in measurement_type.lower() or 'particle' in measurement_type.lower() or 'peening_media' in measurement_type.lower()):
        dist = {'nm': 1e-3, 'µm': 1, 'μm': 1, 'um': 1, 'mm': 1e3, 'cm': 1e4, 'm': 1e6}
        if u in dist:
            return val * dist[u], 'µm', uncertainty * dist[u], ''
        
        micron_re = re.compile(r'µm\s*\([^)]*\)')
        if micron_re.match(u):
            return val, 'µm', uncertainty, ''

    ### Deposition efficiency
    if measurement_type in ['resultsValues.Deposition_Efficiency']:
        deposition_efficiency_generic = {'%': 1, 'fraction': 1}
        if u in deposition_efficiency_generic:
            return val * deposition_efficiency_generic[u], '%', uncertainty * deposition_efficiency_generic[u], ''
        
        deposition_efficiency_vol = {'vol.% (ceramic retention)': 1}
        if u in deposition_efficiency_vol:
            return val * deposition_efficiency_vol[u], 'invalid', uncertainty * deposition_efficiency_vol[u], 'vol % deposition efficiency'

    if 'elastic_modulus' in measurement_type.lower():
        elastic_modulus = {'gpa': 1, 'mpa': 1e-3, 'ksi': 0.0068947573}
        if u in elastic_modulus:
            return val * elastic_modulus[u], 'GPa', uncertainty * elastic_modulus[u], ''
        
        ### Special case -- hand-validated (GPa was read as %)
        if u == '%' and val == 61.5:
            return val, 'GPa', uncertainty, ''

    ### hardness load
    if 'hardness_test_load' in measurement_type.lower():
        # print(f'Examining: {measurement_type}, {u}')
        # Hand-validated misreading corrections
        u = u.replace('kfg', 'kgf')
        const_grav = 9.80665
        load_case_sensitive = {
            'N': 1/const_grav, 'n': 1/const_grav,
            'mN': 1e-3/const_grav, 'mn': 1e-3/const_grav,
            'µN': 1e-6/const_grav, 'µn': 1e-6/const_grav, 'μN': 1e-6/const_grav,
                }
        if case_sensitive_u in load_case_sensitive:
            return val * load_case_sensitive[case_sensitive_u], 'kgf', uncertainty * load_case_sensitive[case_sensitive_u], ''
        load = {
                'mg': 1e-6, 'mgf': 1e-6, 'mg-f': 1e-6,
                'g': 1e-3, 'gf': 1e-3, 'g-f': 1e-3,
                'kg': 1, 'kgf': 1, 'kg-f': 1,
                }
        if u in load:
            return val * load[u], 'kgf', uncertainty * load[u], ''
        if u in ['gpa', 's', 's^-1']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Microhardness
    if measurement_type.lower() == "experimentalproperties.deposit_microhardness" or 'microforging_particle_hardness' in measurement_type.lower() or measurement_type.lower() == 'resultsvalues.deposit_microhardness' or measurement_type.lower() == "experimentalproperties.deposit_nanohardness" or measurement_type.lower() == 'resultsvalues.deposit_nanohardness':
        # Hand-validated misreading corrections
        u = u.replace('hvo.3', 'hv0.3')
        # "hvs" was hand-validated as misread of HV<sub>5</sub> (vickers w 5kgf)

        const_grav = 9.80665
        ### Vickers hardness
        ### Try matching unit WITH numerical load value
        hv_num_str = r'^(?:hv|vh)n?[ _]?(\d+(?:\.\d+)?)([a-z]+)?'
        hv_num_re = re.compile(hv_num_str)
        match = hv_num_re.match(u)
        if match:
            load_value = float(match.group(1))
            if match.group(2) is not None:
                load_units = match.group(2)
            # If no units, default to kgf if load value is less than 50, otherwise default to gf
            else:
                if load_value < 50: load_units = 'kgf'
                else:               load_units = 'gf'
            return val, f'HN_vickers_load_{str(load_value)}_units_{load_units}', uncertainty, ''
        ### For other patterns, just return the hardness value
        vickers_unspecified = {'hvn', 'vhn', 'hv','hn_vickers', 'vickers', 'hv (kgf/mm^2)'}
        # vickers_specified = {'hvs', 'hvo.3', 'hv0.3', 'hv1kgf', 'vhn_1000g'}
        if u in vickers_unspecified:
            return val, 'HN_vickers', uncertainty, ''

        generic_hardness = {'n/mm2': 1/const_grav, 'mpa': 1/const_grav, 'gpa': 1e3/const_grav,
        'kg/mm2': 1, 'kg/mm^2': 1, 'kgf/mm2': 1, 'kgf/mm^2': 1, 'gf/µm^2': 1e3}
        if u in generic_hardness:
            return val * generic_hardness[u], 'invalid', uncertainty * generic_hardness[u], 'incompatible unit: kg/mm^2'

        knoop_hardness = {'knoop': 1, 'khn': 1, 'hk': 1, 'knoop hk': 1, 'hk knoop': 1, 'hkn': 1, 'kgf/mm^2 (knoop)': 1, 'hk500': 1, 'knoop hk500': 1, 'hk knoop500': 1, 'hkn500': 1}
        if u in knoop_hardness:
            return val * knoop_hardness[u], 'invalid', uncertainty * knoop_hardness[u], 'incompatible unit: HN_knoop'
        
        rockwell_hardness = {'hr (rockwell)': 1}
        if u in rockwell_hardness:
            return val * rockwell_hardness[u], 'invalid', uncertainty * rockwell_hardness[u], 'incompatible unit: HN_rockwell_unspecified'

        rockwell_a_hardness = {'hra': 1}
        if u in rockwell_a_hardness:
            return val * rockwell_a_hardness[u], 'invalid', uncertainty * rockwell_a_hardness[u], 'incompatible unit: HN_rockwell_a'
        
        rockwell_b_hardness = {'hrb': 1}
        if u in rockwell_b_hardness:
            return val * rockwell_b_hardness[u], 'invalid', uncertainty * rockwell_b_hardness[u], 'incompatible unit: HN_rockwell_b'

        rockwell_c_hardness = {'hrc': 1}
        if u in rockwell_c_hardness:
            return val * rockwell_c_hardness[u], 'invalid', uncertainty * rockwell_c_hardness[u], 'incompatible unit: HN_rockwell_c'
        
        rockwell_15t_hardness = {'hr15t': 1}
        if u in rockwell_15t_hardness:
            return val * rockwell_15t_hardness[u], 'invalid', uncertainty * rockwell_15t_hardness[u], 'incompatible unit: HN_rockwell_15t'
        
        brinell_hardness = {'hbw': 1, 'hb5': 1, 'hb': 1}
        if u in brinell_hardness:
            return val * brinell_hardness[u], 'invalid', uncertainty * brinell_hardness[u], 'incompatible unit: HN_brinell'
        
        martens_hardness = {'n/mm2 (martens)': 1, 'hm': 1}
        if u in martens_hardness:
            return val * martens_hardness[u], 'invalid', uncertainty * martens_hardness[u], 'incompatible unit: HN_martens'

    ### Stress
    if measurement_type in ['resultsValues.Yield_Strength', 'resultsValues.Ultimate_Tensile_Strength']:
        pressure = {'mpa': 1.0, 'gpa': 1e3, 'ksi': 6.89476, 'n/mm2': 1}
        if u in pressure:
            return val * pressure[u], 'MPa', uncertainty * pressure[u], ''

    ### Strain
    if measurement_type in ['resultsValues.Elongation_at_Break']:
        eng_strain = {'%': 1, 'fraction': 1, 'mm/mm': 1, 'el%': 1, 'el. %': 1, '%el': 1, 'el. %' : 1, 'el %' : 1, '% el' : 1, 'e8%': 1}
        if u in eng_strain:
            return val * eng_strain[u], 'el%', uncertainty * eng_strain[u], ''
    
    ### Temperature
    if 'temperature' in measurement_type.lower():
        if u in ('◦c', 'c', '°c', 'celsius'):
            return val, '°C', uncertainty, ''
        
        if u in ('k', 'kelvin'):
            return val - 273.15, '°C', uncertainty, ''
        
        if u in ('f', '°f', 'fahrenheit'):
            return (val - 32) * 5 / 9, '°C', uncertainty * 5 / 9, ''
        
        if u in ['500']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'
        
    ### Wavelength
    if 'wavelength' in measurement_type.lower():
        wavelength = {'nm': 1, 'µm': 1e3, 'μm': 1e3, 'um': 1e3, 'mm': 1e6, 'cm': 1e7, 'm': 1e9}
        if u in wavelength:
            return val * wavelength[u], 'nm', uncertainty * wavelength[u], ''

    ### Surface roughness
    if 'surface_roughness' in measurement_type.lower():
        roughness = {'µm': 1, 'μm': 1, 'um': 1, 'mm': 1e3, 'cm': 1e4, 'm': 1e6, 
                     'µin': 0.0254}
        if u in roughness:
            return val * roughness[u], 'µm', uncertainty * roughness[u], ''

    ### Impact velocity
    if 'impact_velocity_ratio' in measurement_type.lower():
        impact_velocity_ratio = {'%': 1, 'dimensionless': 1, '–': 1, 'vp/vcr': 1}
        if u in impact_velocity_ratio:
            return val * impact_velocity_ratio[u], 'dimensionless', uncertainty * impact_velocity_ratio[u], ''

    ### Rotation speed
    if 'rotation_speed' in measurement_type.lower():
        rotation_speed = {'r/s': 60, 'r s⁻¹': 60, 'r s-1': 60, 'rs-1': 60, 'rev/min': 1, 'r.p.m.': 1,
                          'rpm': 1, 'r/min': 1, 'r/min⁻¹': 1, 'r/min-1': 1, 'r/min⁻¹': 1, 'r/min-1': 1,
                          'r/h': 1/60, 'r h⁻¹': 1/60, 'r h-1': 1/60, 'rh-1': 1/60,
                          'rad/min': 1/(2*np.pi), 'rad/s': 1/(2*np.pi)*60,
                          }
        if u in rotation_speed:
            return val * rotation_speed[u], 'rpm', uncertainty * rotation_speed[u], ''
        if u in ['%']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Particle volume fraction
    if 'volume_fraction' in measurement_type.lower():
        particle_volume_fraction = {'vol%': 1, 'vol.%': 1, '%': 1}
        if u in particle_volume_fraction:
            return val * particle_volume_fraction[u], 'dimensionless', uncertainty * particle_volume_fraction[u], ''
        if u in ['wt%']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Reduction ratio
    if 'reduction_ratio' in measurement_type.lower():
        reduction_ratio = {'%': 1}
        if u in reduction_ratio:
            return val * reduction_ratio[u], '%', uncertainty * reduction_ratio[u], ''

    ### Pass count
    if 'pass_count' in measurement_type.lower():
        pass_count = {'pass': 1, 'passes': 1}
        if u in pass_count:
            return val * pass_count[u], 'passes', uncertainty * pass_count[u], ''

    ### Strain rate
    if 'strain_rate' in measurement_type.lower():
        
        ### Strain rate without length units
        strain_rate = {
            's⁻¹': 1, 's^-1': 1, 's-1': 1, 's⁻¹': 1, 's^-1': 1, 's-1': 1, 's⁻¹': 1, 
            's^-1': 1, 's-1': 1, '1/s': 1, 'mm/mm/s': 1, 
            'min^-1': 1/60, 'mm/mm/min': 1/60,
        }
        if u in strain_rate:
            return val * strain_rate[u], 's^-1', uncertainty * strain_rate[u], ''
        
        ### Strain rate with length units
        strain_rate_w_len = {
            'mm/min': 1/60, 'mm/s': 1, 'mm·s^-1': 1, 'mm·s-1': 1, 'mm·s⁻¹': 1, 'mm·s^-1': 1, 'mm·s-1': 1,
            'cm/min': 10/60, 'cm/s': 10, 'cm·s^-1': 10, 'cm·s-1': 10, 'cm·s⁻¹': 10, 'cm·s^-1': 10, 'cm·s-1': 10,
            'in/min': 0.4233333333333333,
        }
        ### If the strain rate has length units, we will try to convert to dimensionless units using the gauge length
        if u in strain_rate_w_len:
            # 1) Get gauge length from additional_info_dict (may be missing/blank)
            gauge_length_value = additional_info_dict.get(
                'experimentalProperties.Tensile_Specimen_Gauge_Length_Value', None
            )
            gauge_length_units = additional_info_dict.get(
                'experimentalProperties.Tensile_Specimen_Gauge_Length_Units', ''
            )

            # 2) Safely check if it's a usable number
            try:
                gl_float = float(gauge_length_value)
                gl_is_valid = not np.isnan(gl_float)
            except (TypeError, ValueError):
                gl_is_valid = False

            # 3) If valid, ensure it’s in mm and convert to dimensionless s^-1
            if gl_is_valid:
                gauge_length_value = gl_float
                gauge_length_value, gauge_length_units, _, _ = convert_unit(
                    gauge_length_value,
                    gauge_length_units,
                    measurement_type='experimentalProperties.Tensile_Specimen_Gauge_Length',
                    pass_thru_msg=''
                )
                if gauge_length_units == 'mm' and gauge_length_value not in (0, None):
                    conversion_factor = 1.0 / gauge_length_value
                    return (
                        val * strain_rate_w_len[u] * conversion_factor,
                        's^-1',
                        uncertainty * strain_rate_w_len[u] * conversion_factor,
                        ''
                    )

            # No usable gauge length: keep current behavior (flag as invalid)
            return np.nan, 'invalid', np.nan, f'{case_sensitive_u}: abs. strain rate units but no valid gauge length'
        if u in ['n/s (cross-head)', 'mpa/s', 'n/s', 'mm/mon']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'


    ### Speed
    if ('speed' in measurement_type.lower() or 'velocity' in measurement_type.lower()) \
        and 'rotation' not in measurement_type.lower() and 'ratio' not in measurement_type.lower():
        speed = {'m/s': 1e3, 'mm/s': 1, 'cm/s': 1e1, 'm/min': 1e3/60, 'mm/min': 1/60, 'cm/min': 1e1/60, 
                'm/h': 1e3/3600, 'mm/h': 1/3600, 'cm/h': 1e1/3600,
                'm min⁻¹': 16.666666666666668, 'm min^-1': 16.666666666666668,
                'm min⁻¹ (fill)': 16.666666666666668,
                'mm/sec': 1.0, 'mm s^-1': 1.0, 'mm s-1': 1.0, 'mm s⁻¹': 1.0,
                'mm min-1': 0.016666666666666666, 'mm min^-1': 0.016666666666666666, 'mm min⁻¹': 0.016666666666666666,
                'in/min': 0.4233333333333333, 'in./min': 0.4233333333333333, 'ipm': 0.4233333333333333,
                'km/s': 1e6
                 }
        if u in speed:
            return val * speed[u], 'mm/s', uncertainty * speed[u], ''
        if u in ['min^-1']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Angle
    if 'angle' in measurement_type.lower() :
        angle = {'°': 1, 'deg': 1, 'degree': 1, 'degrees': 1, '◦': 1}
        if u in angle:
            return val * angle[u], 'deg', uncertainty * angle[u], ''

    ### Pressure
    if 'pressure' in measurement_type.lower():
        pressure = {'mpa': 1.0, 'pa': 1e-6, 'gpa': 1e3, 'kpa': 1e-3, 'n/mm2': 1, 'bar': 0.1, 'barg': 0.1, 'bars': 0.1,
                    'ksi': 6.89476,'psi': 0.0068947573, 'psig': 0.0068947573, 
                    'atm': 0.101325, 'torr': 0.00013332236842105263,
                    'kg/mm2': 9.80665, 'kg/cm2': 0.0980665}
        if u in pressure:
            return val * pressure[u], 'MPa', uncertainty * pressure[u], ''

    ### Flow rate
    if 'flow_rate' in measurement_type.lower():
        flow_rate = {'slm': 1, 'l/min': 1, 'l min^-1': 1, 'slpm': 1, 'lpm': 1, 'l min⁻¹': 1, 'l min-1': 1, 
                     'nl/min': 1, 'nl min⁻¹': 1, 'nl min-1': 1, 'l/min (main gas)': 1,
                     'l h⁻¹': 60, 'l h-1': 60, 'l/h': 60, 'l h⁻¹': 60, 'l h-1': 60, 'l/h': 60,
                     'l/s': 1/60, 'l s⁻¹': 1/60, 'l s-1': 1/60,
                     'ml/min': 1/60000, 'ml/s': 1/3600000, 'ml s⁻¹': 1/3600000, 'ml s-1': 1/3600000,
                     'm^3/h': 1e3/60, 'm^3/m': 1e3, 'm^3/s': 1e3*60,
                     'm^3/hr': 1e3/60, 'm^3/min': 1e3, 'm^3/sec': 1e3*60,
                     'm3/h': 1e3/60, 'm3/min': 1e3, 'm3/s': 1e3*60,
                     'm³/h': 1e3/60, 'm³/m': 1e3, 'm³/s': 1e3*60,
                     'm³/hr': 1e3/60, 'm³/min': 1e3, 'm³/sec': 1e3*60,
                     'm3/hr': 1e3/60, 'm^3/hour': 1e3/60, 'm^3/hour': 1e3/60, 'm3/hour': 1e3/60,
                     'nm3/h': 1e3/60, 
                     'dm3/min': 1,
                     'l min⁻¹ (centre purge); 4 l min⁻¹ powder flow': 1,
        }

        if u in flow_rate:
            return val * flow_rate[u], 'L/min', uncertainty * flow_rate[u], ''
        if u in ['m/min','m/h', 'm/s','m/h','m2/h','g/min', 'mm/s', 't/min','m2/min']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Mass feed rate
    if 'powder_feed_rate' in measurement_type.lower():
        mass_feed_rate = {
            'g/h': 60, 'g h-1': 60, 'g h⁻¹': 60, 'g h^-1': 60,
            'g/hr': 60, 'g hr-1': 60, 'g hr⁻¹': 60, 'g hr^-1': 60,
            'g/m': 1, 'g m-1': 1, 'g m⁻¹': 1, 'g m^-1': 1, 'g·min^-1': 1, 'gr/min': 1,
            'g/min': 1, 'g min-1': 1, 'g min⁻¹': 1, 'g min^-1': 1, 'g min⁻1': 1,
            'g/min (pocket; contour value not stated)': 1,
            'g/s': 1/60, 'g s-1': 1/60, 'g s⁻¹': 1/60, 'g s^-1': 1/60, 'g·s^-1': 1/60,
            'g/sec': 1/60, 'g sec-1': 1/60, 'g sec⁻¹': 1/60, 'g sec^-1': 1/60,
            'kg/h': 1e3*60, 'kg h-1': 1e3*60, 'kg h⁻¹': 1e3*60, 'kg h^-1': 1e3*60,
            'kg/hr': 1e3*60, 'kg hr-1': 1e3*60, 'kg hr⁻¹': 1e3*60, 'kg hr^-1': 1e3*60,
            'kg/m': 1e3, 'kg m-1': 1e3, 'kg m⁻¹': 1e3, 'kg m^-1': 1e3,
            'kg/min': 1e3, 'kg min-1': 1e3, 'kg min⁻¹': 1e3, 'kg min^-1': 1e3,
            'kg/s': 1e3/60, 'kg s-1': 1e3/60, 'kg s⁻¹': 1e3/60, 'kg s^-1': 1e3/60,
            'kg/sec': 1e3/60, 'kg sec-1': 1e3/60, 'kg sec⁻¹': 1e3/60, 'kg sec^-1': 1e3/60,
            'mg/s': 1e-3/60, 'mg s-1': 1e-3/60, 'mg s⁻¹': 1e-3/60, 'mg s^-1': 1e-3/60,
                          }
        if u in mass_feed_rate:
            return val * mass_feed_rate[u], 'g/min', uncertainty * mass_feed_rate[u], ''
        unparsable = {'cm^3 h^-1', 'cm^3/min', '%', 'rpm', 'r/min', 'g mm⁻¹','m^3/h','mm/s','flow meter reading','cm3/min','m3/h','nm3/h', 'rad/min','slm', 'ml/min','fmr','rev/min','l/min','m^3/hour','m/h'}
        if u in unparsable:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Time/duration
    if 'dwell_time' in measurement_type.lower():
        time = {'ms': 1e-3, 's': 1, 'min': 60, 'h': 3600,
                'seconds': 1, 'minutes': 60, 'hours': 3600,
                'sec': 1, 'min': 60, 'hr': 3600,
                'second': 1, 'minute': 60, 'hour': 3600,
                }
        if u in time:
            return val * time[u], 's', uncertainty * time[u], ''
        if u in ['mn/min']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'
        
    if 'duration' in measurement_type.lower() or 'cool_down_time' in measurement_type.lower():
        duration = {'s': 1/3600, 'sec': 1/3600, 'seconds': 1/3600,
                    'm': 1/60, 'min': 1/60, 'minutes': 1/60, 'mins': 1/60,
                    'h': 1, 'hr': 1, 'hours': 1, 'hour': 1,
                    'days': 24*3600, 'day': 24*3600,
                    }
        if u in duration:
            return val * duration[u], 's', uncertainty * duration[u], ''
        if u in ['%', 'impacts']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Power
    if 'power_level' in measurement_type.lower() or 'laser_power' in measurement_type.lower():
        power_level = {'w': 1, 'kw': 1e3,
                       }
        if u in power_level:
            return val * power_level[u], 'W', uncertainty * power_level[u], ''
        if u in ['w (reduced 1 w per layer)', 'w (reduced 2 w per layer)', 'w (reduced 3 w per layer)', 'w (reduced 4 w per layer)', 'w (reduced 5 w per layer)', 'gw/cm2']:
            return np.nan, 'invalid', np.nan, f'incompatible unit: {case_sensitive_u}'

    ### Length
    if 'diameter' in measurement_type.lower() or 'length' in measurement_type.lower() or 'standoff_distance' in measurement_type.lower() or 'step_size' in measurement_type.lower() or 'thickness' in measurement_type.lower() or 'width' in measurement_type.lower() or 'mean_size' in measurement_type.lower():
        length = {'mm': 1, 'mm (spot size)': 1, 'mm (defocused)': 1, 'mm (hatch spacing)': 1, 
                  'mm (hatch distance sx)': 1,
                  'µm': 1e-3, 'μm': 1e-3, 'um': 1e-3, 'cm': 1e-1, 'm': 1e3,
                  'in': 25.4, 'inch': 25.4, 'inches': 25.4, 'in.': 25.4}
        if u in length:
            return val * length[u], 'mm', uncertainty * length[u], ''
        ### If provided area, convert to diameter
        if u in ['mm²', 'mm^2'] and 'diameter' in measurement_type.lower():
            
            def conv_area_to_diameter(area):
                """Convert area to diameter, assuming circular cross-section"""
                return 2*np.sqrt(area / np.pi)
            
            ### Special case for Article_1036.md
            if pass_thru_msg == 'Article_1036.md':
                if 'nozzle_throat' in measurement_type.lower():
                    print('Custom conversion for Article_1036.md: nozzle_throat')
                    return conv_area_to_diameter(6), 'mm', conv_area_to_diameter(uncertainty), ''
                if 'nozzle_exit' in measurement_type.lower():
                    print('Custom conversion for Article_1036.md: nozzle_exit')
                    return conv_area_to_diameter(40), 'mm', conv_area_to_diameter(uncertainty), ''

            print(f'Converting area to diameter: {val} {u} -> {conv_area_to_diameter(val)} mm | {pass_thru_msg}')
            return conv_area_to_diameter(val), 'mm', conv_area_to_diameter(uncertainty), ''

    ### Notify of new unknown units
    else:
        print(f'Unknown unit: {case_sensitive_u} for value: {val} of type: {measurement_type}\t{pass_thru_msg}')
    return val, 'invalid', uncertainty, f'unknown unit: {case_sensitive_u}'


Z_VAL_CUTOFF = 3    

suspicious_value_upper_limit = {
    'resultsValues.Porosity':                   25,
    'resultsValues.Elastic_Modulus':            300,
    'resultsValues.Yield_Strength':             2000,
    'resultsValues.Ultimate_Tensile_Strength':  20000,
    'resultsValues.Elongation_at_Break':        60,
    'resultsValues.Deposit_Microhardness':      1000,
    'resultsValues.Deposit_Nanohardness':       1000,
    'resultsValues.Deposition_Efficiency':      100,
    'resultsValues.Grain_Size':                 250,
    'experimentalProperties.Carrier_Gas_Flow_Rate': 1e4,
    'experimentalProperties.Carrier_Gas_Pressure': 10,
    'experimentalProperties.Carrier_Gas_Temperature': 2000,
    'experimentalProperties.Cold_Spray_Spot_Diameter': 10,
    'experimentalProperties.Deposit_Microhardness_Test_Dwell_Time': 30,
    'experimentalProperties.Deposit_Microhardness_Test_Load': 100,
    'experimentalProperties.Deposit_Nanohardness_Test_Dwell_Time': 30,
    'experimentalProperties.Deposit_Nanohardness_Test_Load': 100,
    'experimentalProperties.Deposit_Post_First_Treatment_Cool_Down_Time': 120,
    'experimentalProperties.Deposit_Post_First_Treatment_Duration': 120,
    'experimentalProperties.Deposit_Post_First_Treatment_Temperature': 2000,
    'experimentalProperties.Deposit_Post_Second_Treatment_Cool_Down_Time': 120,
    'experimentalProperties.Deposit_Post_Second_Treatment_Duration': 120,
    'experimentalProperties.Deposit_Post_Second_Treatment_Temperature': 2000,
    'experimentalProperties.Deposit_Post_Third_Treatment_Cool_Down_Time': 120,
    'experimentalProperties.Deposit_Post_Third_Treatment_Duration': 120,
    'experimentalProperties.Deposit_Post_Third_Treatment_Temperature': 2000,
    'experimentalProperties.Deposit_Tensile_Test_Strain_Rate': 1,
    'experimentalProperties.Friction_Stir_Probe_Diameter': 20,
    'experimentalProperties.Friction_Stir_Rotation_Speed': np.inf,
    'experimentalProperties.Friction_Stir_Shoulder_Diameter': np.inf,
    'experimentalProperties.Friction_Stir_Tilt_Angle': np.inf,
    'experimentalProperties.Friction_Stir_Travel_Speed': np.inf,
    'experimentalProperties.HIP_Duration': np.inf,
    'experimentalProperties.HIP_Pressure': np.inf,
    'experimentalProperties.HIP_Temperature': np.inf,
    'experimentalProperties.Hot_Rolling_Pass_Count': np.inf,
    'experimentalProperties.Hot_Rolling_Reduction_Ratio': np.inf,
    'experimentalProperties.Hot_Rolling_Speed': np.inf,
    'experimentalProperties.Hot_Rolling_Temperature': 2000,
    'experimentalProperties.Impact_Velocity_Ratio': np.inf,
    'experimentalProperties.Inter_Layer_Dwell_Time': 600,
    'experimentalProperties.Laser_Beam_Diameter': 10,
    'experimentalProperties.Laser_Power': 1e4,
    'experimentalProperties.Laser_Power_Level': 1e4,
    'experimentalProperties.Laser_Spray_Spot_Temperature': 2000,
    'experimentalProperties.Laser_Wavelength': 2000,
    'experimentalProperties.Microforging_Particle_Hardness': 1000,
    'experimentalProperties.Microforging_Particle_Mean_Size': 10,
    'experimentalProperties.Microforging_Particle_STD_Size': 10,
    'experimentalProperties.Microforging_Particle_Volume_Fraction': 100,
    'experimentalProperties.Nozzle_Exit_Diameter': 100,
    'experimentalProperties.Nozzle_Expansion_Section_Length': 1000,
    'experimentalProperties.Nozzle_Length': 1000,
    'experimentalProperties.Nozzle_Throat_Diameter': 100,
    'experimentalProperties.Particle_Critical_Velocity': np.inf,
    'experimentalProperties.Particle_Impact_Velocity': np.inf,
    'experimentalProperties.Particle_Maximum_Velocity': np.inf,
    'experimentalProperties.Particle_Minimum_Velocity': np.inf,
    'experimentalProperties.Particle_Temperature_At_Impact': np.inf,
    'experimentalProperties.Particle_Velocity': np.inf,
    'experimentalProperties.Peening_Duration': np.inf,
    'experimentalProperties.Peening_Media_Mean_Size': np.inf,
    'experimentalProperties.Peening_Media_STD_Size': np.inf,
    'experimentalProperties.Peening_Media_Volume_Fraction': np.inf,
    'experimentalProperties.Peening_Pressure': np.inf,
    'experimentalProperties.Powder_Feed_Rate': 1000,
    'experimentalProperties.Powder_Feeder_Rotation_Speed': 60,
    'experimentalProperties.Process_Gas_Flow_Rate': 4000,
    'experimentalProperties.Process_Gas_Pressure': 10,
    'experimentalProperties.Process_Gas_Temperature': 2000,
    'experimentalProperties.Rastering_Speed': 500,
    'experimentalProperties.Rastering_Step_Size': 10,
    'experimentalProperties.Spray_Angle': 90,
    'experimentalProperties.Standoff_Distance': 100,
    'experimentalProperties.Surface_Roughness': 100,
    'experimentalProperties.Tensile_Specimen_Gauge_Length': 100,
    'experimentalProperties.Tensile_Specimen_Gauge_Thickness': 10,
    'experimentalProperties.Tensile_Specimen_Gauge_Width': 10,
    'experimentalProperties.Tensile_Test_Temperature': 1000,
    'preSprayedProperties.Powder_Particle_Mean_Size': np.inf,
    'preSprayedProperties.Powder_Particle_STD_Size': np.inf,
    'preSprayedProperties.Powder_Particle_Size_D10': np.inf,
    'preSprayedProperties.Powder_Particle_Size_D50': np.inf,
    'preSprayedProperties.Powder_Particle_Size_D90': np.inf,
    'preSprayedProperties.Powder_Particle_Min_Size': np.inf,
    'preSprayedProperties.Powder_Particle_Max_Size': np.inf,
    'preSprayedProperties.Powder_Pre-treatment_Temperature': np.inf,
    'preSprayedProperties.Powder_Pre-treatment_Duration': np.inf,
    'preSprayedProperties.Powder_Pre-treatment_Cool_Down_Time': np.inf,
    'preSprayedProperties.Substrate_Thickness': np.inf
}
suspicious_value_lower_limit = {
    'resultsValues.Porosity':                   0,
    'resultsValues.Elastic_Modulus':            0,
    'resultsValues.Yield_Strength':             20,
    'resultsValues.Ultimate_Tensile_Strength':  50,
    'resultsValues.Elongation_at_Break':        1,
    'resultsValues.Deposit_Microhardness':      1,
    'resultsValues.Deposit_Nanohardness':       1,
    'resultsValues.Deposition_Efficiency':      1,
    'resultsValues.Grain_Size':                 0.005,
    'experimentalProperties.Carrier_Gas_Flow_Rate': 0,
    'experimentalProperties.Carrier_Gas_Pressure': 0,
    'experimentalProperties.Carrier_Gas_Temperature': 0,
    'experimentalProperties.Cold_Spray_Spot_Diameter': 0,
    'experimentalProperties.Deposit_Microhardness_Test_Dwell_Time': 0,
    'experimentalProperties.Deposit_Microhardness_Test_Load': 0,
    'experimentalProperties.Deposit_Nanohardness_Test_Dwell_Time': 0,
    'experimentalProperties.Deposit_Nanohardness_Test_Load': 0,
    'experimentalProperties.Deposit_Post_First_Treatment_Cool_Down_Time': 0,
    'experimentalProperties.Deposit_Post_First_Treatment_Duration': 0,
    'experimentalProperties.Deposit_Post_First_Treatment_Temperature': 0,
    'experimentalProperties.Deposit_Post_Second_Treatment_Cool_Down_Time': 0,
    'experimentalProperties.Deposit_Post_Second_Treatment_Duration': 0,
    'experimentalProperties.Deposit_Post_Second_Treatment_Temperature': 0,
    'experimentalProperties.Deposit_Post_Third_Treatment_Cool_Down_Time': 0,
    'experimentalProperties.Deposit_Post_Third_Treatment_Duration': 0,
    'experimentalProperties.Deposit_Post_Third_Treatment_Temperature': 0,
    'experimentalProperties.Deposit_Tensile_Test_Strain_Rate': 0,
    'experimentalProperties.Friction_Stir_Probe_Diameter': 0,
    'experimentalProperties.Friction_Stir_Rotation_Speed': 0,
    'experimentalProperties.Friction_Stir_Shoulder_Diameter': 0,
    'experimentalProperties.Friction_Stir_Tilt_Angle': 0,
    'experimentalProperties.Friction_Stir_Travel_Speed': 0,
    'experimentalProperties.HIP_Duration': 0,
    'experimentalProperties.HIP_Pressure': 0,
    'experimentalProperties.HIP_Temperature': 0,
    'experimentalProperties.Hot_Rolling_Pass_Count': 0,
    'experimentalProperties.Hot_Rolling_Reduction_Ratio': 0,
    'experimentalProperties.Hot_Rolling_Speed': 0,
    'experimentalProperties.Hot_Rolling_Temperature': 0,
    'experimentalProperties.Impact_Velocity_Ratio': 0,
    'experimentalProperties.Inter_Layer_Dwell_Time': 0,
    'experimentalProperties.Laser_Beam_Diameter': 0,
    'experimentalProperties.Laser_Power': 0,
    'experimentalProperties.Laser_Power_Level': 0,
    'experimentalProperties.Laser_Spray_Spot_Temperature': 0,
    'experimentalProperties.Laser_Wavelength': 0,
    'experimentalProperties.Microforging_Particle_Hardness': 0,
    'experimentalProperties.Microforging_Particle_Mean_Size': 0,
    'experimentalProperties.Microforging_Particle_STD_Size': 0,
    'experimentalProperties.Microforging_Particle_Volume_Fraction': 0,
    'experimentalProperties.Nozzle_Exit_Diameter': 0,
    'experimentalProperties.Nozzle_Expansion_Section_Length': 0,
    'experimentalProperties.Nozzle_Length': 0,
    'experimentalProperties.Nozzle_Throat_Diameter': 0,
    'experimentalProperties.Particle_Critical_Velocity': 0,
    'experimentalProperties.Particle_Impact_Velocity': 0,
    'experimentalProperties.Particle_Maximum_Velocity': 0,
    'experimentalProperties.Particle_Minimum_Velocity': 0,
    'experimentalProperties.Particle_Temperature_At_Impact': 0,
    'experimentalProperties.Particle_Velocity': 0,
    'experimentalProperties.Peening_Duration': 0,
    'experimentalProperties.Peening_Media_Mean_Size': 0,
    'experimentalProperties.Peening_Media_STD_Size': 0,
    'experimentalProperties.Peening_Media_Volume_Fraction': 0,
    'experimentalProperties.Peening_Pressure': 0,
    'experimentalProperties.Powder_Feed_Rate': 0,
    'experimentalProperties.Powder_Feeder_Rotation_Speed': 0,
    'experimentalProperties.Process_Gas_Flow_Rate': 0,
    'experimentalProperties.Process_Gas_Pressure': 0,
    'experimentalProperties.Process_Gas_Temperature': 0,
    'experimentalProperties.Rastering_Speed': 0,
    'experimentalProperties.Rastering_Step_Size': 0,
    'experimentalProperties.Spray_Angle': 0,
    'experimentalProperties.Standoff_Distance': 0,
    'experimentalProperties.Surface_Roughness': 0,
    'experimentalProperties.Tensile_Specimen_Gauge_Length': 0,
    'experimentalProperties.Tensile_Specimen_Gauge_Thickness': 0,
    'experimentalProperties.Tensile_Specimen_Gauge_Width': 0,
    'experimentalProperties.Tensile_Test_Temperature': 0,
    'preSprayedProperties.Powder_Particle_Mean_Size': 0,
    'preSprayedProperties.Powder_Particle_STD_Size': 0,
    'preSprayedProperties.Powder_Particle_Size_D10': 0,
    'preSprayedProperties.Powder_Particle_Size_D50': 0,
    'preSprayedProperties.Powder_Particle_Size_D90': 0,
    'preSprayedProperties.Powder_Particle_Min_Size': 0,
    'preSprayedProperties.Powder_Particle_Max_Size': 0,
    'preSprayedProperties.Powder_Pre-treatment_Temperature': 0,
    'preSprayedProperties.Powder_Pre-treatment_Duration': 0,
    'preSprayedProperties.Powder_Pre-treatment_Cool_Down_Time': 0,
    'preSprayedProperties.Substrate_Thickness': 0
}

PRINT_MODE = "stats"  # default; overridden by CLI


def _first_nonnull(series_like):
    """
    Return the first value that isn't None/NaN/empty-string.
    Works with scalars, lists, dicts, numpy scalars, etc.
    """
    if isinstance(series_like, (pd.Series, np.ndarray, list, tuple)):
        values = series_like
    else:
        values = [series_like]

    for x in values:
        if x is None or x is pd.NA:
            continue
        if isinstance(x, float) and np.isnan(x):
            continue
        if isinstance(x, str) and x.strip() == "":
            continue
        return x
    return None



def _flatten_articles_to_experiment_rows(input_path: str) -> pd.DataFrame:
    """
    Turn the article-level JSON into one row per experiment with dotted keys.
    Preserves article-level fields by prefixing them with 'article.' and
    flattens the 'metadata' dict to 'article.metadata.<Key>'.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for art in (data or []):
        ex = art.get("extractedText", {})
        exps = ex.get("Experiments", []) if isinstance(ex, dict) else []
        if not isinstance(exps, list):
            continue

        # Article-level fields (everything except extractedText/metadata)
        article_prefix = {}
        for k, v in art.items():
            if k in ("extractedText", "metadata"):
                continue
            article_prefix[f"article.{k}"] = v

        # Flatten metadata
        meta = art.get("metadata", {})
        if isinstance(meta, dict):
            for mk, mv in meta.items():
                article_prefix[f"article.metadata.{mk}"] = mv
        else:
            article_prefix["article.metadata"] = meta

        for exp in exps:
            row = {}
            row.update(article_prefix)
            # Also keep plain filename/title for convenience
            row["filename"] = art.get("filename")
            row["title"] = art.get("title")
            for cat in ("preSprayedProperties", "experimentalProperties", "resultsValues"):
                block = exp.get(cat, {})
                if isinstance(block, dict):
                    for k, v in block.items():
                        row[f"{cat}.{k}"] = v
            rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure the strain-rate helper columns exist
    for col in [
        "experimentalProperties.Tensile_Specimen_Gauge_Length_Value",
        "experimentalProperties.Tensile_Specimen_Gauge_Length_Units",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    return df

def _unflatten_experiment_rows_to_articles(df: pd.DataFrame, include_qc: bool = False):
    """
    Rebuild the nested article -> Experiments structure from a flat DataFrame.

    Input columns expected:
      - 'filename' (used to group rows into articles)
      - 'title' (optional; first non-null kept per article)
      - 'article.*' (article-level fields, incl. 'article.metadata.*')
      - '<cat>.*' where <cat> in {'preSprayedProperties','experimentalProperties','resultsValues'}
      - Optional QC columns: start with 'zscore_' or 'outlier_' (attached to each experiment if include_qc=True)

    Returns:
      List[dict] shaped like the original nested JSON.
    """
    if df is None or df.empty:
        return []

    # Article-level fields
    art_cols = [c for c in df.columns if c.startswith("article.")]
    meta_prefix = "article.metadata."

    # Experiment blocks
    cats = ("preSprayedProperties", "experimentalProperties", "resultsValues")
    cat_cols = {
        cat: [
            c for c in df.columns
            if c.startswith(f"{cat}.") and not c.endswith("_Pass_Thru_Msg")
        ]
        for cat in cats
    }


    # QC columns
    qc_prefixes = ("zscore_", "outlier_")
    qc_cols = [c for c in df.columns if any(c.startswith(p) for p in qc_prefixes)] if include_qc else []

    articles = []

    # Group by article (filename). If you prefer (filename,title), change here.
    for (fname,), g in df.groupby(["filename"], dropna=False):
        # Ensure JSON-safe filename (avoid NaN)
        safe_fname = None
        if not (isinstance(fname, float) and np.isnan(fname)):
            safe_fname = fname

        art = {"filename": safe_fname}
        if "title" in g.columns:
            art["title"] = _first_nonnull(g["title"])

        # Copy article.* (excluding metadata.* which we rebuild separately)
        for c in art_cols:
            if c.startswith(meta_prefix):
                continue
            key = c[len("article."):]
            if c in g:
                art[key] = _first_nonnull(g[c])

        # Rebuild metadata dict from article.metadata.*
        meta = {}
        for c in art_cols:
            if not c.startswith(meta_prefix):
                continue
            mk = c[len(meta_prefix):]
            if c in g:
                meta[mk] = _first_nonnull(g[c])
        if meta:
            art["metadata"] = meta

        # Build experiments list
        exps = []
        for _, r in g.iterrows():
            exp = {}
            for cat in cats:
                sub = {}
                for c in cat_cols[cat]:
                    sub[c[len(cat) + 1:]] = r.get(c, None)
                exp[cat] = sub

            if qc_cols:
                qc = {}
                for col in qc_cols:
                    v = r.get(col, None)
                    # Make JSON-serializable if it's a numpy scalar
                    if isinstance(v, np.generic):
                        v = v.item()
                    qc[col] = v
                if qc:
                    exp["qc"] = qc

            exps.append(exp)

        art["extractedText"] = {"Experiments": exps}
        articles.append(art)

    return articles

import os
import json

def standardize_cold_spray(
    input_path: str,
    output_path: str,
    print_mode: str = "A",  # "A"/"stats" or "B"/"verbose"
    categories=("resultsValues", "experimentalProperties", "preSprayedProperties"),
    output_format: str = "flat",  # "flat" or "nested"
    include_qc: bool = False,     # only used if output_format == "nested"
):
    """
    End-to-end:
      1) read nested article JSON and flatten to experiment rows
      2) parse value/unit using parse_values_and_units (assumed defined above)
      3) compute zscores/outlier flags using your thresholds
      4) save either flat JSON (records) or nested back to article shape

    Returns the cleaned DataFrame (flat).
    """
    # -------- setup print mode --------
    global PRINT_MODE
    PRINT_MODE = "stats" if print_mode in ("A", "stats") else "verbose"

    # -------- load & flatten ----------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}\nResolved: {os.path.abspath(input_path)}")

    print("\n### cold_spray ###\n########################################################")
    df = _flatten_articles_to_experiment_rows(input_path)
    print(f"Found {len(df):,} experiments after flattening.")

    # -------- clean & convert ---------
    print("\nCleaning cold_spray dataset...")
    CATEGORIES_TO_CLEAN = [c.strip() for c in categories if c]

    target_cols = [c for c in df.columns if c.split(".")[0] in CATEGORIES_TO_CLEAN]

    # exclude keys you never want to process
    EXCLUDE_KEYS = {
        "preSprayedProperties.Powder_Blend_Ratio_Standardized",
    }
    keys_to_clean = [
        k for k in (c.replace("_Value", "") for c in target_cols if c.endswith("_Value"))
        if k not in EXCLUDE_KEYS
    ]
    print(f"{len(keys_to_clean)} keys to clean")

    if PRINT_MODE == "verbose":
        print(keys_to_clean)

    for key in keys_to_clean:
        print(f"\n{key}")
        # parse values/units (your function is assumed present)
        parsed = list(df.apply(parse_values_and_units, args=(key,), axis=1))
        (
            parsed_vals,
            parsed_units,
            parsed_uncertainties,
            err_messages_value,
            err_messages_unit,
            pass_thru_msgs,
        ) = zip(*parsed) if parsed else ([], [], [], [], [], [])

        df[f"{key}_Value"] = parsed_vals
        df[f"{key}_Units"] = parsed_units
        unc_series = pd.Series(parsed_uncertainties, dtype="object")
        df[f"{key}_Uncertainty"] = unc_series.where(pd.notna(unc_series), "")

        # df[f"{key}_ErrorValue"] = err_messages_value
        # df[f"{key}_ErrorUnits"] = err_messages_unit
        # df[f"{key}_Pass_Thru_Msg"] = pass_thru_msgs

        # Printouts
        if PRINT_MODE == "verbose":
            counts = pd.Series(parsed_units, dtype="object").value_counts(dropna=False)
            print(counts.to_string())
        else:
            mapped = sum(u != "invalid" for u in parsed_units)
            unmapped = sum(u == "invalid" for u in parsed_units)
            print(f"\tUnits mapped: {mapped:,} | Units unmapped: {unmapped:,}")

        # Outlier detection
        df[f"zscore_{key}"] = zscore(pd.to_numeric(df[f"{key}_Value"], errors="coerce"), nan_policy="omit")
        df[f"outlier_z1_{key}"] = df[f"zscore_{key}"] > 1
        df[f"outlier_z2_{key}"] = df[f"zscore_{key}"] > 2
        df[f"outlier_z3_{key}"] = df[f"zscore_{key}"] > 3

        # Threshold-based suspicious flags (safe defaults if key absent)
        upper = suspicious_value_upper_limit.get(key, np.inf)
        lower = suspicious_value_lower_limit.get(key, -np.inf)
        vals_num = pd.to_numeric(df[f"{key}_Value"], errors="coerce")
        suspicious_rows = (vals_num > upper) | (vals_num < lower)
        df[f"outlier_suspicious_{key}"] = suspicious_rows

        # Combine
        df[f"outlier_{key}"] = df[f'outlier_z{Z_VAL_CUTOFF}_{key}'] | df[f"outlier_suspicious_{key}"]

        # Defragment copy (avoids pandas fragmentation warning)
        df = df.copy()

        if PRINT_MODE == "verbose":
            print(f"\t{int(df[f'outlier_{key}'].sum())} rows with outliers")

    # -------- save output -------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    if output_format == "nested":
        articles = _unflatten_experiment_rows_to_articles(df, include_qc=include_qc)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
    else:
        # flat
        df.to_json(output_path, orient="records", indent=2)

    print(f"\nSaved cleaned cold_spray dataset to: {output_path}\n")
    return df

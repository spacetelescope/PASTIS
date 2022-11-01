from pastis.config import CONFIG_PASTIS


STANDARD_SECTIONS = ['local', 'telescope', 'numerical', 'zernikes', 'calibration', 'dm_objects']
OBSERVATORY_SECTIONS = ['JWST', 'RST', 'HiCAT', 'LUVOIR', 'LUVOIR-B']


def test_main_sections():
    """Check that all main sections exist."""

    for section in STANDARD_SECTIONS+OBSERVATORY_SECTIONS:
        exists = section in CONFIG_PASTIS
        assert exists, f'Section {section} does not exist in configfile.'


def test_data_paths():
    """Check that all required data paths exist."""

    data_keys = ['local_data_path', 'webbpsf_data_path']
    for key in data_keys:
        assert CONFIG_PASTIS.has_option('local', key), f"[local] section has no key '{key}'"


def test_telescope():
    """Check that all required telescope keys exist."""

    telescope_keys = ['name']
    for key in telescope_keys:
        assert CONFIG_PASTIS.has_option('telescope', key), f"[telescope] section has no key '{key}'"


def test_observatory_parameters():
    """Check that all observatory sections have all necessary keys.
    If using a local configfile like recommended in the documentation, this test
    will only catch missing observatory parameters on custom planet sections if the test is
    run locally.
    """

    observatory_params = ['calibration_aberration', 'valid_range_lower', 'valid_range_upper', 'nb_subapertures',
                          'lambda', 'sampling']
    all_sections = CONFIG_PASTIS.sections()

    # First test the observatory sections that are included by default
    for sec in OBSERVATORY_SECTIONS:
        for key in observatory_params:
            assert CONFIG_PASTIS.has_option(sec, key), f"'{sec}' section has no key '{key}'"

    # Then test any additional observatory sections
    for sec in all_sections:
        if sec in STANDARD_SECTIONS+OBSERVATORY_SECTIONS:
            pass
        else:
            for key in observatory_params:
                assert CONFIG_PASTIS.has_option(sec, key), f"'{sec}' section has no key '{key}'"

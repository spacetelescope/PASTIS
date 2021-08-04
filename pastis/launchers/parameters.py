'''
Create class to help launcher and keep some parameter in attributs
'''

import logging

import pastis.e2e_simulators.telescopes
from pastis.matrix_generation.matrix_building_numerical import MatrixIntensity
from pastis.matrix_generation.matrix_from_efields import MatrixEfield
from pastis.config import CONFIG_PASTIS

log = logging.getLogger()


def gen_method(dir=''):
    '''
    Select the right matrix generator
    '''
    method = CONFIG_PASTIS.get('generation', 'method')
    param = parameters()
    param.def_saves()
    param.log_off()

    if method == 'intensity':
        gen = MatrixIntensity(initial_path=dir, param=param)
    elif method == 'Efield':
        gen = MatrixEfield(initial_path=dir, param=param)
    else:
        error_msg = f"{method} inside config.ini file is not a valid generation method!" \
                    f"excepted intensity or Efield"
        log.error(error_msg)
        raise ValueError(error_msg)
    log.info(f'Start matrix generation with {method} method')
    return gen


class parameters():

    def __int__(self, initial_path=''):

        super().__init__()

    def def_telescope(self):
        self.instrument = CONFIG_PASTIS.get('telescope', 'name')
        if self.instrument == 'RST':
            self.telescope = pastis.e2e_simulators.telescopes.RST()
        elif self.instrument == 'JWST':
            self.telescope = pastis.e2e_simulators.telescopes.JWST()
        elif self.instrument == 'LUVOIR':
            self.telescope = pastis.e2e_simulators.telescopes.LUVOIRA()
        else:
            error_msg = f"{self.instrument} inside config.ini file is not a valid telescope name!" \
                        f"excepted JWST, RST, ATLAST, HiCAT or LUVOIR"
            log.error(error_msg)
            raise ValueError(error_msg)

        return self.telescope

    def def_saves(self):
        self.savepsfs = CONFIG_PASTIS.getboolean('save_data', 'save_psfs')
        self.saveopds = CONFIG_PASTIS.getboolean('save_data', 'save_opds')
        self.saveefields = CONFIG_PASTIS.getboolean('save_data', 'save_efields')
        self.save_coro_floor = CONFIG_PASTIS.getboolean('save_data', 'save_coro_floor')
        self.return_coro_simulator = CONFIG_PASTIS.getboolean('save_data', 'coro_simulator')

    def log_off(self):
        mplfm_logger = logging.getLogger('matplotlib.font_manager')
        mplcb_logger = logging.getLogger('matplotlib.colorbar')
        mplt_logger = logging.getLogger('matplotlib.ticker')
        mplbe_logger = logging.getLogger('matplotlib.backends')

        mplfm_logger.setLevel(logging.WARNING)
        mplcb_logger.setLevel(logging.WARNING)
        mplt_logger.setLevel(logging.WARNING)
        mplbe_logger.setLevel(logging.WARNING)

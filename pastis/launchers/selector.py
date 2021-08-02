import pastis.e2e_simulators.telescopes
import logging
from pastis.config import CONFIG_PASTIS

log = logging.getLogger()

class paramaters():

    def telescope(self):
        self.instrument = CONFIG_PASTIS.get('telescope', 'name')
        if self.instrument == 'RST':
            self.telescope = pastis.e2e_simulators.telescopes.RST()
        if self.instrument == 'JWST':
            self.telescope = pastis.e2e_simulators.telescopes.RST()
        else:
            error_msg = f"{self.instrument} inside config.ini file is not a valid telescope name!" \
                        f"excepted JWST, RST, ATLAST, HiCAT or LUVOIR"
            log.error(error_msg)
            raise ValueError(error_msg)

        return self.telescope

    def saves(self):
        self.savepsfs = CONFIG_PASTIS.getboolean('save_data', 'save_psfs')
        self.saveopds = CONFIG_PASTIS.getboolean('save_data', 'save_opds')
        self.save_coro_floor = CONFIG_PASTIS.getboolean('save_data', 'save_coro_floor')
        self.return_coro_simulator = CONFIG_PASTIS.getboolean('save_data', 'coro_simulator')

import numpy as _np
from os import path as _path
from datetime import datetime as _dt
import logging as _logging
from ruamel.yaml.comments import CommentedMap as _CommentedMap


from godot.cosmos.util import load_yaml as _load_yaml
from godot.core.tempo import Epoch as _Epoch

class Config:
    def __init__(self, filename_universe='universe.yml', filename_trajectory='trajectory.yml'):
        if _path.isfile(filename_universe):
            self.universe = _load_yaml(filename_universe)
        else:
            _logging.error('YAML file for universe missing')
        if _path.isfile(filename_trajectory):
            self.trajectory = _load_yaml(filename_trajectory)
        else:
            _logging.error('YAML file for universe missing')

    CART_ELEMENTS = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
    KEP_ELEMENTS = ['sma', 'ecc', 'inc', 'ran', 'aop', 'tan']

    def set_geopotential_degree_order(self, degree: int, order: int):
        self.universe['sphericalHarmonics'][0]['config']['degree'] = int(degree)
        self.universe['sphericalHarmonics'][0]['config']['order'] = int(order)

    def set_object_mass(self, mass: float):
        self.universe['spacecraft'][0]['mass'] = str(float(mass)) + ' kg'

    def set_object_srp_area(self, area: float):
        self.universe['spacecraft'][0]['srp']['area'] = str(float(area)) + ' m^2'

    def set_object_srp_coeff(self, coeff: float):
        self.universe['spacecraft'][0]['srp']['cr'] = str(float(coeff))

    def set_object_drag_area(self, area: float):
        self.universe['spacecraft'][0]['drag']['area'] = str(float(area)) + ' m^2'

    def set_object_drag_coeff(self, coeff: float):
        self.universe['spacecraft'][0]['drag']['cr'] = str(float(coeff))

    def get_epoch(self):
        return _Epoch(self.trajectory['timeline'][1]['epoch'])

    def set_epoch(self, epoch: _Epoch):
        self.trajectory['timeline'][1]['epoch'] = str(epoch)

    def get_epoch_past(self):
        return _Epoch(self.trajectory['timeline'][0]['point']['epoch'])

    def set_epoch_past(self, epoch: _Epoch):
        self.trajectory['timeline'][0]['point']['epoch'] = str(epoch)
    
    def get_epoch_future(self):
        return _Epoch(self.trajectory['timeline'][2]['point']['epoch'])
    
    def set_epoch_future(self, epoch: _Epoch):
        self.trajectory['timeline'][2]['point']['epoch'] = str(epoch)

    def set_state_cart(self, x):
        # Remove any previous set of elements
        self.trajectory['timeline'][1]['state'][0]['value'] = _CommentedMap()
        # Add cartesian elements
        for i, element in enumerate(self.CART_ELEMENTS):
            if 'pos' in element:
                value = str(float(x[i])) + ' km'
            else:
                value = str(float(x[i])) + ' km/s'
            self.trajectory['timeline'][1]['state'][0]['value'][element] = value

    def set_state_kep(self, x):
        # Remove any previous set of elements
        self.trajectory['timeline'][1]['state'][0]['value'] = _CommentedMap()
        # Add keplerian elements
        for i, element in enumerate(self.KEP_ELEMENTS):
            if 'sma' in element:
                value = str(float(x[i])) + ' km'
            elif 'ecc' in element:
                value = float(x[i])
            else:
                value = str(float(x[i])) + ' deg'
            self.trajectory['timeline'][1]['state'][0]['value'][element] = value
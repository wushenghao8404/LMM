from .uniformcycletsc import UniformCycleTSC
from .websterstsc import WebstersTSC
from .maxpressuretsc import MaxPressureTSC
from .sotltsc import SOTLTSC
from .nextphaserltsc import NextPhaseRLTSC
from .nextdurationrltsc import NextDurationRLTSC


def tsc_factory(tsc_type, tl, args, netdata, conn):
    if tsc_type == 'webster':
        return WebstersTSC(conn, tl, netdata, args['r'], args['y'],
                           args['g_min'], args['c_min'],
                           args['c_max'], sat_flow=args['sat_flow'],
                           update_freq=args['update_freq'])
    elif tsc_type == 'sotl':
        return SOTLTSC(conn, tl, netdata, args['r'], args['y'],
                       args['g_min'], args['theta'], args['omega'],
                       args['mu'] )
    elif tsc_type == 'uniform':
        return UniformCycleTSC(conn, tl, netdata, args['r'], args['y'], args['g_min'])
    elif tsc_type == 'maxpressure':
        return MaxPressureTSC(conn, tl, netdata, args['r'], args['y'],
                              args['g_min'] )
    else:
        #raise not found exceptions
        assert 0, 'Supplied traffic signal control argument type '+str(tsc_type)+' does not exist.'

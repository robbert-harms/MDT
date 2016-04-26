from mdt.models.compartments import CompartmentConfig, CLCodeFromInlineString

__author__ = 'Francisco.Lagos'

#From protocol, if the signal is SE, we can setup TM = 0 in all the volumes, which returns to the standard SE signal decay


class ExpT1ExpT2STEAM(CompartmentConfig):

    name = 'ExpT1ExpT2STEAM'
    cl_function_name = 'cmExpT1ExpT2STEAM'
    parameter_list = ('TM', 'TE', 'T1', 'T2')
    cl_code = CLCodeFromInlineString("""
        return 1/2.0 * exp(-TE / T2) * exp(-TM / T1);
    """)
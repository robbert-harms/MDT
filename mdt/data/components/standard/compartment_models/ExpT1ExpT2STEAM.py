from mdt.models.compartments import CompartmentConfig, CLCodeFromInlineString

__author__ = 'Francisco.Lagos'

# For SE volumes: FA2 = FA3 and FA2 (and FA3) has to be HALF OF THE FLIP ANGLE USED IN THE PROTOCOL!, also SEf = 0 and TM = 0.
# For STE volumes: SEf = 1.

class ExpT1ExpT2STEAM(CompartmentConfig):

    name = 'ExpT1ExpT2STEAM'
    cl_function_name = 'cmExpT1ExpT2STEAM'
    parameter_list = ('SEf', 'FA1', 'FA2', 'FA3', 'TM', 'TE', 'T1', 'T2')
    cl_code = CLCodeFromInlineString("""
        return pow(0.5,SEf) * sin(FA1) * sin(FA2) * sin(FA3) * exp(-TE / T2) * exp(-TM / T1);
    """)

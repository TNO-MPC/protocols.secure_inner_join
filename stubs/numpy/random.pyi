from numpy.random._generator import Generator as NumpyGenerator
from numpy.random.bit_generator import BitGenerator as NumpyBitGenerator
from randomgen.common import BitGenerator as RandomGenBitGenerator

class Generator(NumpyGenerator):
    # numpy.random._generator.Generator requires numpy.random.BitGenerator, PCG32 is a randomgen.common.BitGenerator
    # They're mostly compatible.
    def __init__(
        self, bit_generator: NumpyBitGenerator | RandomGenBitGenerator
    ) -> None: ...

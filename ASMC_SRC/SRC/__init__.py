# This file is part of ASMC, developed by Pier Francesco Palamara.

# ASMC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# ASMC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with ASMC.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import collections
from .asmc import DecodingModeOverall, asmc

ASMCReturnValues = collections.namedtuple(
    "ASMCReturnValues",
    "sumOverPairs sumOverPairs00 sumOverPairs01 sumOverPairs11")


def to_array(x):
    a = list(x)
    if a:
        return np.array(a)
    else:
        return None


def run(haps_file_root, decoding_quant_file, out_file_root="",
        mode=DecodingModeOverall.array, jobs=0,
        job_index=0, skip_csfs_distance=0,
        compress=False, use_ancestral=False,
        posterior_sums=False, major_minor_posterior_sums=False):
    ret = asmc(haps_file_root=haps_file_root,
               decoding_quant_file=decoding_quant_file,
               mode=mode, jobs=jobs, job_index=job_index,
               skip_csfs_distance=skip_csfs_distance,
               compress=compress, use_ancestral=use_ancestral,
               posterior_sums=posterior_sums,
               major_minor_posterior_sums=major_minor_posterior_sums)
    return ASMCReturnValues(
        sumOverPairs=to_array(ret.sumOverPairs),
        sumOverPairs00=to_array(ret.sumOverPairs00),
        sumOverPairs01=to_array(ret.sumOverPairs01),
        sumOverPairs11=to_array(ret.sumOverPairs11))
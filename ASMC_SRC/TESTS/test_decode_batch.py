import unittest
import numpy as np
import msprime
from asmc import (
    HMM,
    DecodingQuantities,
    DecodingParams,
    Data,
    makePairObs,
)
import os
import shutil
from contextlib import contextmanager
from timeit import default_timer


def btime_default_f(time_in_seconds):
    print("Time elapsed (seconds):", time_in_seconds)


@contextmanager
def btime(f=btime_default_f):
    """Brian timer.
    Times a block of code and applies function f to the resulting time in seconds.
    Inspired by https://stackoverflow.com/a/30024601.
    """
    start = default_timer()
    yield
    end = default_timer()
    time_in_seconds = end - start
    f(time_in_seconds)



class TestASMCDecode(unittest.TestCase):

    def setUp(self):
        self.random_seed = 1
        self.sample_size = 300
        self.length = 1e6
        self.recomb_rate = 1.2e-8
        self.min_maf = 0

    def test_foo(self):
        simulation = msprime.simulate(
          Ne=2e4,
          sample_size=self.sample_size,
          mutation_rate = 1.65e-8,
          length=self.length,
          recombination_rate=self.recomb_rate,
          random_seed=self.random_seed
        )

        if self.min_maf == 0:
            mode = "sequence"
        elif self.min_maf > 0 and self.min_maf < 1:
            mode = "array"
        else:
            raise ValueError("MAF filter should lie in [0, 1)")

        if simulation.num_samples % 2 != 0:
            raise ValueError("Even number of samples expected")

        single_run = "1"  # holdover from earlier, so we don't need to change relate.sh
        ASMC_TMP_DIR = "tmp_" + mode + "/"
        # logging.info("Using temporary directory " + ASMC_TMP_DIR)
        os.makedirs(ASMC_TMP_DIR, exist_ok=True)

        out_path = ASMC_TMP_DIR + str(single_run) + ".hap"
        site_positions = []
        with open(out_path, 'w') as out_file:
            last = -1
            for variant in simulation.variants():
                af = np.mean(variant.genotypes)
                if af < self.min_maf or 1 - af < self.min_maf:
                    continue
                pos = int(variant.site.position)
                if pos <= last:
                    pos = last + 1
                site_positions.append(pos)
                last = pos
                row_list = ['1', '.', str(pos), '0', '1']
                row_list += [str(entry) for entry in variant.genotypes]
                # print(row_list)
                # break
                out_file.write(' '.join(row_list))
                out_file.write('\n')
        site_positions = np.array(site_positions)
        out_path = ASMC_TMP_DIR + str(single_run) + ".samples"
        with open(out_path, 'w') as out_file:
            out_file.write('\t'.join(["ID_1", "ID_2", "missing"]) + '\n')
            out_file.write('\t'.join(["0", "0", "0"]) + '\n')
            for i in range(simulation.num_samples // 2):
                out_file.write('\t'.join(["sample_" + str(i), "sample_" + str(i), "0"]) + '\n')

        # In this case, just multiply by the rate
        assert self.recomb_rate is not None
        chrom_string = "chr"
        # Write out result
        out_path = ASMC_TMP_DIR + str(single_run) + ".map"
        with open(out_path, "w") as out_file:
            for site_pos in site_positions:
                site_cm = (self.recomb_rate * 1e8) * site_pos / 1e6
                out_file.write('\t'.join([chrom_string, "SNP_" + str(site_pos),
                    str(site_cm), str(site_pos)]) + '\n')

        # start to set up ASMC object
        haps_file_root = ASMC_TMP_DIR + str(single_run)
        decoding_quant_file = "FILES/DECODING_QUANTITIES" \
            "/30-100-2000.decodingQuantities.gz"

        with btime(lambda x: print("Time building HMM (seconds):", x)):
            sequence_length = Data.countHapLines(haps_file_root)
            if mode == "sequence":
                params = DecodingParams(haps_file_root, decoding_quant_file,
                    compress=True, skipCSFSdistance=float('nan'),
                    decodingModeString="sequence", useAncestral=False)
            elif mode == "array":
                params = DecodingParams(haps_file_root, decoding_quant_file,
                    compress=False, skipCSFSdistance=0,
                    decodingModeString="array", useAncestral=False)
            else:
                raise ValueError("Unrecognized mode, must be one of sequence or array")
            decoding_quantities = DecodingQuantities(decoding_quant_file)
            data = Data(haps_file_root, sequence_length,
                        decoding_quantities.CSFSSamples, params.foldData,
                        params.usingCSFS)
            hmm = HMM(data, decoding_quantities, params, not params.noBatches, 1)
            # shutil.rmtree(ASMC_TMP_DIR)

        expected_times = np.array(decoding_quantities.expectedTimes)
        def decode_pair_slow(i, j):
            pair_obs = makePairObs(data.individuals[i // 2], i % 2 + 1,
                                   data.individuals[j // 2], j % 2 + 1)
            # Note: posterior is 69 x phys_pos.shape[0], where 69 is the number of discretizations
            decode_result = hmm.decode(pair_obs)
            posterior = np.array(decode_result)
            a = decoding_quantities.expectedTimes[0] # use this to capture decoding_quantities
            # Calculate MAP and posterior mean
            posterior_map = expected_times[posterior.argmax(axis=0)]
            posterior_mean = np.sum(posterior * expected_times[:, np.newaxis], axis=0)
            return site_positions, posterior_map, posterior_mean

        def decode_pair(i, j):
            pair_obs = makePairObs(data.individuals[i // 2], i % 2 + 1,
                                   data.individuals[j // 2], j % 2 + 1)
            decode_result = hmm.decodeSummarize(pair_obs)
            # return site_positions, np.array(decode_result[0]), np.array(decode_result[1])
            return site_positions, decode_result[0], decode_result[1]  # this performed the same?

        def decode_pair_eigen(i, j):
            pair_obs = makePairObs(data.individuals[i // 2], i % 2 + 1,
                                   data.individuals[j // 2], j % 2 + 1)
            decode_result = hmm.decodeSummarizeEigen(pair_obs)
            return site_positions, decode_result[0], decode_result[1]

        def decode_batch(i):
            pair_obs_list = []
            for j in range(0, i):
                pair_obs = makePairObs(data.individuals[i // 2], i % 2 + 1,
                                       data.individuals[j // 2], j % 2 + 1)
                pair_obs_list.append(pair_obs)
            decode_result = hmm.decodeSummarizeBatch(pair_obs_list, print_time=True)
            # let's have decode_result be (2 * batch_size) x length

            tmrca_map = decode_result[:len(pair_obs_list)]
            tmrca_mean = decode_result[len(pair_obs_list):]
            return tmrca_map, tmrca_mean

        # i = 3  # this works well for 1e4
        i = 39  # this works great for 1e5

        with btime(lambda x: print("Time batch (seconds):", x)):
            with btime(lambda x: print("Time in C++ batch (seconds):", x)):
                tmrca_map, tmrca_mean = decode_batch(i)
            indices_batch = np.argmin(tmrca_map, axis=0)
            times_batch = np.min(tmrca_map, axis=0)

        with btime(lambda x: print("Time regular (seconds):", x)):
            posterior_phys_pos = None
            tmrca_map = None
            tmrca_mean = None
            for j in range(0, i):
                res = decode_pair(i, j)  # a tuple containing the return values
                if posterior_phys_pos is None:
                    posterior_phys_pos = res[0]
                if tmrca_map is None:
                    tmrca_map = np.zeros((i, len(posterior_phys_pos)))
                    tmrca_mean = np.zeros((i, len(posterior_phys_pos)))
                tmrca_map[j] = res[1]
                tmrca_mean[j] = res[2]

            indices_nobatch = np.argmin(tmrca_map, axis=0)
            times_nobatch = np.min(tmrca_map, axis=0)

        with btime(lambda x: print("Time Eigen (seconds):", x)):
            posterior_phys_pos = None
            tmrca_map = None
            tmrca_mean = None
            for j in range(0, i):
                # res = decode_pair_slow(i, j)  # a tuple containing the return values
                res = decode_pair_eigen(i, j)  # a tuple containing the return values
                if posterior_phys_pos is None:
                    posterior_phys_pos = res[0]
                if tmrca_map is None:
                    tmrca_map = np.zeros((i, len(posterior_phys_pos)))
                    tmrca_mean = np.zeros((i, len(posterior_phys_pos)))
                tmrca_map[j] = res[1]
                tmrca_mean[j] = res[2]

            indices_eigen = np.argmin(tmrca_map, axis=0)
            times_eigen = np.min(tmrca_map, axis=0)

        with btime(lambda x: print("Time compare (seconds):", x)):
            self.assertTrue(np.all(np.equal(indices_nobatch, indices_eigen)))
            self.assertTrue(np.allclose(times_nobatch, times_eigen, rtol=0, atol=1e-8))

            print(np.sum(1 - np.equal(indices_nobatch, indices_batch)))
            # self.assertTrue(np.all(np.equal(indices_nobatch, indices_batch)))
            # self.assertTrue(np.allclose(times_nobatch, times_batch, rtol=0, atol=1e-8))
            self.assertTrue(np.allclose(times_nobatch, times_batch, rtol=0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

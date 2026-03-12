from thousand_tasks.training.act_bn_reaching.dataset import BCDataset
from torch.utils.data.distributed import DistributedSampler

from typing import Optional, Iterator
import torch
import math
import numpy as np

# def get_verbose_print(verbose: bool):
#     none_lambda = lambda x: None
#     if verbose:
#         return print
#     return none_lambda

class DistributedCustomSampler(DistributedSampler):
    def __init__(self, dataset: BCDataset, batch_size: int,
                 chuncks2load: int = None,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0, drop_last: bool = False,
                 verbose: bool = False,
                 use_distributed: bool = True) -> None:

        if seed == -1:
            seed = np.random.randint(500, 1000)
        self.verbose = verbose

        if use_distributed:
            super().__init__(dataset, num_replicas, rank, False, seed, drop_last)
        else:
            super().__init__(dataset, 1, 0, False, seed, drop_last)

        self.dataset: BCDataset
        self.batch_size = batch_size
        self.n_chunks = self.dataset.n_chunks
        self.printv(f"\n\n[SAMPLER] Total chunks: {self.n_chunks}")
        dps_per_chunk = self.dataset.dp_per_chunk
        self.printv(f"[SAMPLER] Dps per chunk: {dps_per_chunk}")
        chuncks2load = batch_size if chuncks2load is None else chuncks2load
        self.printv(f"[SAMPLER] Chunks to load: {chuncks2load}")
        self.dps_per_load = chuncks2load * dps_per_chunk
        self.chuncks2load = chuncks2load

        if self.drop_last and self.n_chunks % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (self.n_chunks - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.n_chunks / self.num_replicas)  # type: ignore[arg-type]

        self.printv(f"\n[SAMPLER] Chunks in GPU: {self.num_samples}")

        # to make sure # loaded chunks is always batch_size
        self.batches_per_load = self.dps_per_load // batch_size
        rounded_increment = chuncks2load - (self.num_samples % chuncks2load)
        self.num_samples += rounded_increment

        self.printv(f"[SAMPLER] Chunks in GPU rounded: +{rounded_increment}"
                    f" --> {self.num_samples}")
        self.num_loadings = self.num_samples // chuncks2load
        self.total_size = self.num_samples * self.num_replicas

        self.dataset.batch_size = batch_size
        self.dataset.dataset_length = len(self) * self.num_replicas

    def __len__(self) -> int:
        return self.num_loadings * self.batches_per_load * self.batch_size

    def printv(self, statement: str):
        if self.verbose:
            print(statement)

    def chunk_id_yielder(self, indices: torch.Tensor):
        for snum in range(self.num_loadings):
            passed_indeces = snum * self.num_replicas * self.chuncks2load
            other_rank_indeces = self.rank * self.chuncks2load
            start_id = passed_indeces + other_rank_indeces
            yield indices[
                start_id : (start_id+self.chuncks2load)
            ]

    def __iter__(self) -> Iterator:
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.n_chunks, generator=g).tolist()  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        self.printv(f"\n\n[SAMPLER] All chunk ids:{len(indices)}\n{indices}\n")

        dps2use_per_load = self.batch_size * self.batches_per_load
        self.printv(f"[SAMPLER] All loaded dps: {self.dps_per_load}")
        self.printv(f"[SAMPLER] dps to use: {dps2use_per_load}")
        self.printv(f"[SAMPLER] batches in load: {dps2use_per_load / self.batch_size}")
        self.printv(f"[SAMPLER] Useless dps: {self.dps_per_load - dps2use_per_load} / {self.batch_size}\n")
        chunk_gen = self.chunk_id_yielder(indices)
        for chunk_n, chunk_ids in enumerate(chunk_gen):
            self.printv(f"[SAMPLER] Chunk {chunk_n}: {chunk_ids}")

        datapoints_per_chunk = self.dataset.dp_per_chunk
        chunk_gen = self.chunk_id_yielder(indices)
        dps2use_per_load = self.batch_size * self.batches_per_load
        for chunk_ids in chunk_gen:
            self.printv(f"\n\n\n\n[SAMPLER] Chunk ids:\n{chunk_ids}\n")
            self.dataset.load_chunks(chunk_ids)
            data_point_ids = np.arange(0, self.dps_per_load)
            np.random.shuffle(data_point_ids)
            self.printv(f"[SAMPLER] Data Points ids:\n{data_point_ids}\n")
            for dp_id in data_point_ids[:dps2use_per_load]:
                yield dp_id

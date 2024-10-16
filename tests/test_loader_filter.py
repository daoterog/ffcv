import logging
import os
from dataclasses import replace
from multiprocessing import cpu_count
from tempfile import NamedTemporaryFile
from typing import Callable

import numpy as np
import torch as ch
from assertpy import assert_that
from test_writer import DummyDataset
from torch.utils.data import Dataset

from ffcv.fields import BytesField, FloatField, IntField
from ffcv.fields.basics import FloatDecoder
from ffcv.loader import Loader
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.reader import Reader
from ffcv.transforms.ops import ToTensor
from ffcv.writer import DatasetWriter

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


class Doubler(Operation):

    def generate_code(self) -> Callable:
        def code(x, dst):
            dst[:] = x * 2
            return dst

        return code

    def declare_state_and_memory(self, previous_state: State):
        return (
            previous_state,
            AllocationQuery(
                previous_state.shape, previous_state.dtype, previous_state.device
            ),
        )


def test_basic_simple():
    length = 600
    batch_size = 8
    with NamedTemporaryFile() as handle:
        file_name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(file_name, {"index": IntField(), "value": FloatField()})

        writer.from_indexed_dataset(dataset)

        Compiler.set_enabled(True)

        loader = Loader(
            file_name,
            batch_size,
            num_workers=min(5, cpu_count()),
            seed=17,
            pipelines={
                "value": [FloatDecoder(), Doubler(), ToTensor()],
            },
        )

        def cond(value):
            value = value[0]
            result = value < 1 and value >= 0.5
            return result

        filtered = loader.filter("value", cond)

        assert_that(len(filtered)).is_greater_than(0)
        for index, values in filtered:
            assert_that(values.shape[0]).is_equal_to(batch_size)
            assert_that(((values < 1) & (values >= 0.5)).all()).is_true()

import tempfile
from glob import glob
from os import path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import torch as ch
import webdataset as wds
from test_writer import validate_simple_dataset
from torch.utils.data import Dataset

from ffcv import DatasetWriter
from ffcv.fields import FloatField, IntField
from ffcv.reader import Reader

field_names = ["index", "value.pyd"]


class DummyDataset(Dataset):

    def __init__(self, l):
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index >= self.l:
            raise IndexError()
        return (index, np.sin(index))


def write_webdataset(folder, dataset, field_names):
    pattern = path.join(folder, "dataset-%06d.tar")
    writer = wds.ShardWriter(pattern, maxcount=20)
    with writer as sink:
        for i, sample in enumerate(dataset):
            data = {"__key__": f"sample_{i}"}

            for field_name, value in zip(field_names, sample):
                data[field_name] = value
            sink.write(data)


def pipeline(dataset):
    return dataset.decode().to_tuple(*field_names)


if __name__ == "__main__":
    N = 1007
    dataset = DummyDataset(N)
    with TemporaryDirectory() as temp_directory:
        with NamedTemporaryFile() as handle:
            fname = handle.name
            write_webdataset(temp_directory, dataset, field_names)
            files = glob(path.join(temp_directory, "*"))
            files = list(sorted(files))

            print(fname)
            writer = DatasetWriter(fname, {"index": IntField(), "value": FloatField()})

            writer.from_webdataset(files, pipeline)

            validate_simple_dataset(fname, N, shuffled=False)

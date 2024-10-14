from typing import List, Union

import torch as ch

from ..transforms import ToTensor
from ..transforms.module import ModuleWrapper
from .operation import Operation


class PipelineSpec:

    def __init__(
        self,
        source: Union[str, Operation],
        decoder: Operation = None,
        transforms: List[Operation] = None,
    ):

        self.source = source
        self.decoder = decoder
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.default_pipeline = (
            decoder is None and not transforms and isinstance(source, str)
        )

    def __repr__(self):
        return repr((self.source, self.decoder, self.transforms))

    def __str__(self):
        return self.__repr__()

    def accept_decoder(self, Decoder, output_name):
        if not isinstance(self.source, str) and self.decoder is not None:
            raise ValueError("Source can't be a node and also have a decoder")

        if Decoder is not None:
            # The first element of the operations is a decoder
            if self.transforms and isinstance(self.transforms[0], Decoder):
                self.decoder = self.transforms.pop(0)

            elif self.decoder is None:
                try:
                    self.decoder = Decoder()
                except Exception:
                    msg = f"Impossible to use default decoder for {output_name},"
                    msg += "make sure you specify one in your pipeline."
                    raise ValueError(msg)

        if self.default_pipeline:
            self.transforms.append(ToTensor())

        for i, op in enumerate(self.transforms):
            if isinstance(op, ch.nn.Module):
                self.transforms[i] = ModuleWrapper(op)

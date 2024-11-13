# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class StreamingLLMPress(BasePress):
    """
    Prune a fixed number of KV pairs at the beginning and end of the sequence (https://arxiv.org/abs/2309.17453)
    We keep the first n_sink tokens and the last n_local tokens.
    n_local is computed using the compression ratio.
    """

    compression_ratio: float = 0.0
    n_sink: int = 4

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        q_len = hidden_states.shape[1]
        assert q_len > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        n_pruned = q_len - int(q_len * (1 - self.compression_ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0

        return scores

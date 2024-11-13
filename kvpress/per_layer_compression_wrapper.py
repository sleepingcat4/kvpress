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

import logging
from typing import List

import torch

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


def apply_per_layer_compression(press: BasePress, compression_ratios: List[float]) -> BasePress:
    """
    Apply per-layer compression to a given press object.
    This function wraps the forward hook of the press object to apply per-layer compression.

    Parameters
    ----------
    press : BasePress
        The press object to apply per-layer compression to.
    compression_ratios : Dict[int, float]

    Returns
    -------
    BasePress
        The press object with per-layer compression applied.
    """
    press.compression_ratios = compression_ratios  # type: ignore[attr-defined]
    press.compression_ratio = None

    logger.warning(
        "Per layer compression wrapper is an experimental feature and only works with flash attention. "
        "Please make sure that the model uses flash attention."
    )

    original_forward_hook = press.forward_hook

    def _forward_hook(module: torch.nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        press.compression_ratio = press.compression_ratios[module.layer_idx]  # type: ignore[attr-defined]
        output = original_forward_hook(module, input, kwargs, output)
        press.compression_ratio = None
        return output

    press.forward_hook = _forward_hook  # type: ignore[method-assign]
    return press

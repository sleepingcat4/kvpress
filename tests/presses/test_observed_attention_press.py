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

import torch
from transformers import DynamicCache

from kvpress import ObservedAttentionPress
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


@torch.no_grad()
def test_observed_drops_attention_output(unit_test_model, unit_test_model_output_attention, caplog):  # noqa: F811
    input_ids = unit_test_model.dummy_inputs["input_ids"]
    output = unit_test_model(input_ids, past_key_values=DynamicCache())
    assert output.attentions is None

    input_ids = unit_test_model_output_attention.dummy_inputs["input_ids"]
    attentions = unit_test_model_output_attention(input_ids, past_key_values=DynamicCache()).attentions
    assert all([isinstance(attention, torch.Tensor) for attention in attentions])

    with caplog.at_level(logging.DEBUG):
        press = ObservedAttentionPress(compression_ratio=0.4)
        with press(unit_test_model_output_attention):
            input_ids = unit_test_model_output_attention.dummy_inputs["input_ids"]
            output = unit_test_model_output_attention(input_ids, past_key_values=DynamicCache())

            # There's a slight mismatch in outputs when using a model that has output_attentions=True
            # and removing them in the hook vs. a model that has output_attentions=False
            assert output.attentions == (None, None)

    messages = [record.message for record in caplog.records]
    assert any(["Model will not return attentions in its output to save memory." in message for message in messages])

    press = ObservedAttentionPress(compression_ratio=0.4, output_attentions=True)
    with press(unit_test_model_output_attention):
        input_ids = unit_test_model_output_attention.dummy_inputs["input_ids"]
        output = unit_test_model_output_attention(input_ids, past_key_values=DynamicCache())

        assert all(
            [
                torch.allclose(reference_attention, attention)
                for reference_attention, attention in zip(attentions, output.attentions)
            ]
        )

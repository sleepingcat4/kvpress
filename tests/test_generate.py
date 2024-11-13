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

from kvpress import KnormPress
from tests.fixtures import kv_press_pipeline  # noqa: F401


def test_generate(kv_press_pipeline):  # noqa: F811
    context = "This is a test article. It was written on 2022-01-01."
    press = KnormPress(compression_ratio=0.4)

    # Answer with pipeline
    pipe_answer = kv_press_pipeline(context, press=press, max_new_tokens=10)["answer"]

    # Answer with model.generate
    context += "\n"  # kv press pipeline automatically adds a newline if no chat template
    model = kv_press_pipeline.model
    tokenizer = kv_press_pipeline.tokenizer
    with press(model):
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        generate_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generate_answer = generate_answer[len(context) :]

    assert pipe_answer == generate_answer

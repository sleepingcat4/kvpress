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

import pandas as pd
from datasets import Dataset, load_dataset

MAX_NEW_TOKENS = {
    "gov_report": 1024,
    "summ_screen_fd": 512,
    "qmsum": 512,
    "qasper": 128,
    "narrative_qa": 64,
    "quality": 10,
    "musique": 32,
    "squality": 512,
    "space_digest": 36,
    "book_sum_sort": 256,
}

df_list = []
for task, max_new_tokens in MAX_NEW_TOKENS.items():
    df = load_dataset("tau/zero_scrolls", task, split="test").to_pandas()
    df["context"] = df.apply(lambda x: x["input"][: x["document_end_index"]], axis=1)
    df["question"] = df.apply(lambda x: x["input"][x["document_end_index"] : x["query_end_index"]], axis=1)
    df["answer_prefix"] = df.apply(lambda x: x["input"][x["query_end_index"] :], axis=1).str.strip()
    df["answer"] = ""
    df["task"] = task
    df["max_new_tokens"] = max_new_tokens
    df_list.append(df)

df = pd.concat(df_list)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub(repo_id="zero_scrolls", split="test")

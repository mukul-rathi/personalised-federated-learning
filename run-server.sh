#!/bin/bash

# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

STRATEGY=$1
Q_PARAM=$2
Q_LR=$3

# Start a Flower server
python3.7 server.py \
  --rounds=10 \
  --epochs=5 \
  --sample_fraction=0.5 \
  --min_sample_size=5 \
  --min_num_clients=5 \
 --strategy=$STRATEGY  \
 --qffl_learning_rate=$Q_LR\
 --q_param=$Q_PARAM
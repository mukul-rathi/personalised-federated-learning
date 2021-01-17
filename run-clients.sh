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


SERVER_ADDRESS="[::]:8080"

STRATEGY=$1
ALPHA=$2
BETA=$3
NUM_CLIENTS=$4 
IID_FRACTION=$5

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python3.7 client.py \
      --cid=$i \
      --server_address=$SERVER_ADDRESS \
      --num_partitions=$NUM_CLIENTS \
      --iid_fraction=$IID_FRACTION \
      --strategy=$STRATEGY \
      --alpha=$ALPHA \
      --beta=$BETA \
      --exp_name="${STRATEGY}_alpha=${ALPHA}_beta=${BETA}_federated_${NUM_CLIENTS}_clients" &
done
echo "Started $NUM_CLIENTS clients."


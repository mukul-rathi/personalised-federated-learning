# run both FL server and clients

STRATEGY="qffedAvg"
ALPHA=1e-3
BETA=1e-3
IID_FRACTION=0.1
NUM_CLIENTS=10
Q_LR=1
Q_PARAM=0.01

RUN_PATH="./completed_runs/script_outputs/${STRATEGY}_alpha=${ALPHA}_beta=${BETA}_numclients=${NUM_CLIENTS}_q_lr=${Q_LR}_q_param=${Q_PARAM}"

mkdir $RUN_PATH;
(sleep 2.5 ; ./run-clients.sh $STRATEGY $ALPHA $BETA $NUM_CLIENTS $IID_FRACTION) &  ./run-server.sh $STRATEGY $Q_PARAM $Q_LR;
mv ./runs/* $RUN_PATH
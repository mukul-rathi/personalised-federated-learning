# run both FL server and clients

(trap 'kill 0' SIGINT; (sleep 2.5 ; ./run-clients.sh) &  ./run-server.sh )
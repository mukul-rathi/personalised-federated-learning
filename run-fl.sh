# run both FL server and clients

(trap 'kill 0' SIGINT; ./run-server.sh & ./run-clients.sh)
# personalised-federated-learning

Implementation of per-FedAvg using the Flower federated learning framework, and comparing it against q-FedAvg and FedAvg.

per-FedAvg: https://proceedings.neurips.cc/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf

q-FedAvg: https://arxiv.org/abs/1905.10497

Flower: https://flower.dev/

I implemented this as part of a research mini-project for one of my modules (Principles of ML Systems) in my Master's at Cambridge, where I explored the effect of personalisation in federated learning when clients have heterogeneous data distributions. 

My code is based off a lab assignment run by the module lecturers, (Nic Lane and his research group at Cambridge) - see the initial commits, and license headers for files that I did not substantially modify.

I extended this to implement per-FedAvg, and I slightly modified the implementation of q-FedAvg present in the Flower repository to fix a bug I had. My code is also somewhat messy, apologies! 

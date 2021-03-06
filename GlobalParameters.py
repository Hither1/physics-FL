# =================== general global parameters (EDITABLE) ====================
# number of clients
num_worker = 1

# number of global iteration
global_iteration_num = 20

# number of local epoch per global iteration
local_epoch_num = 10

# aggregation rule
aggregation_rule = 'FedAvg'
# aggregation_rule = 'Krum'
#aggregation_rule = 'FedInv'

# model's device
#device = 'cuda'
device = 'cpu'

# model's type
model='cnn'
# model = 'fc'
# model='lr'

# machine learning task
#task = 'SIR'
task = 'Pendulum'


# number of local batch size
local_batch_size = 200


# =================== general global parameters (EDITABLE) ====================

# =================== global parameters for FedCom (EDITABLE) ====================
commitment_accuracy = 50
# =================== global parameters for FedCom (EDITABLE) ====================


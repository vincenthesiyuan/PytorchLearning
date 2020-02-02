import torch
from torch import nn

"""
Saving & Loading Model Across Devices
save:
    torch.save(model.state_dict(), PATH)

load:
    device = torch.device('cpu')  
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location=device))
"""
# ##########################################################

"""
Save on GPU, Load on GPU
save:
    torch.save(model.state_dict(), PATH)

load:
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model


Note that calling my_tensor.to(device) returns a new copy of my_tensor on GPU.
Therefore, remenber to manually overwrite tensor:
    my_tensor = my_tensor.to(torch.device('cuda'))
"""
# ##########################################

"""
Save on CPU, Load on GPU
save:
    torch.save(model.state_dict(), PATH)

load:
    device = torch.device("cuda")
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
    model.to(device)
    
"""

# ########################################
"""
Saving torch.nn.DataParallel Models
it is a model wrapper that enables parallel GPU utilization.
"""
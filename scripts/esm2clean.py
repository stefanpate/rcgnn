import torch
from src.CLEAN.utils import *
from src.CLEAN.model import LayerNormNet
import os

'''
Params
'''

esm_embed_dir = '../data/swissprot/esm/'
save_to = '../data/swissprot/clean/'
idx = 33 # 32 for eric's embeddings

'''
Helpers
'''

def format_esm(a, idx=idx):
    if type(a) == dict:
        a = a['mean_representations'][idx]
    return a

def load_esm(fn, dir=esm_embed_dir):
    esm = format_esm(torch.load(dir + fn))
    return esm.unsqueeze(0) # Stick a 1 out in front of shape tuple

def get_clean_embeds(train_str, esm_embeds):
    # Torch params
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    
    # Load checkpoints
    # Change to LayerNormNet(512, 256, device, dtype)
    # and rebuild with [python build.py install]
    # if inferencing on model trained w/ supconH loss
    model = LayerNormNet(512, 128, device, dtype) 
    checkpoint = torch.load('../data/swissprot/pretrained/'+ train_str +'.pth', map_location=device) 
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Forward pass
    esm_embeds = torch.cat(esm_embeds).to(device=device, dtype=dtype)
    model_emb = model(esm_embeds)
    return model_emb

'''
Main
'''

print("Loading esm")
esm_embeds = [load_esm(elt) for elt in os.listdir(esm_embed_dir)] # Load esm embeds
print('done')
#fns = os.listdir(esm_embed_dir) # Keep file names
#clean_embeds = get_clean_embeds("split100", esm_embeds)

# Save embeddings
#for i in range(clean_embeds.shape[0]):
        # Must clone a slice of the tensor to save only that slice, not whole tensor
        # Slicing returns a view, not a new object
        # Detach from loss function to get just the vector
#        torch.save(clean_embeds[i].clone().detach(), save_to + fns[i])

#        if i % 1000 == 0:
#             print(f"{i} / {clean_embeds.shape[0]} converted")

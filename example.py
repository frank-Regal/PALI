import torch
from pali.model import ViTModule, Pali


#training data
img = torch.randn(1, 3, 256, 256)
print(img)
print(img.shape)

prompt = torch.randint(0, 256, (1, 1024)) # prompt
print(prompt)

prompt_mask = torch.ones(1, 1024).bool()
print(prompt_mask)
output_text = torch.randint(0, 256, (1, 1024)) #target output text
print(output_text)

#train
img_embeds = ViTModule(
    img, 
    return_embeddings=True
)

loss = Pali(
    prompt,
    output_text,
    mask=prompt_mask,
    src_prepend_embeds=img_embeds # will prepend image embeddings
)

print(loss.backward())


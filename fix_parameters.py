# stole from ziqin
def _freeze_stages(self, model, exclude_key=None):
    """Freeze stages param and norm stats."""
    for n, m in model.named_parameters():
        if exclude_key:
            if isinstance(exclude_key, str):
                if not exclude_key in n:
                    m.requires_grad = False
            elif isinstance(exclude_key, list):
                count = 0
                for i in range(len(exclude_key)):
                    i_layer = str(exclude_key[i])
                    if i_layer in n:
                        count += 1
                if count == 0:
                    m.requires_grad = False
                elif count > 0:
                    print('Finetune layer in backbone:', n)
            else:
                assert AttributeError("Dont support the type of exclude_key!")
        else:
            m.requires_grad = False

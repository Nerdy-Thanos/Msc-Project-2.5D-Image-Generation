import timm
import torch.nn as nn
import pg_modules.discriminator as disc

discrim = disc.ProjectedDiscriminator()
print(discrim)

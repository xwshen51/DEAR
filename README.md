# Disentangled Generative Causal Representation Learning

This repository contains the code for the paper [*Disentangled Generative Causal Representation Learning*](https://arxiv.org/abs/2010.02637).

## Install prerequisites
```
pip install -r requirements.txt
```

## Datasets
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Pendulum (to be released soon)

## Run

- Run DEAR on CelebA:
```
sh run_celeba_smile.sh
```

### Output
This will create a directory `./results/<dataset>/<save_name>` which will contain:

- **model.sav**: a Python distionary containing the generator, encoder, and discriminator.
- **gen.png**: generated images.
- **recon.png**: real images (odd columns) along with the reconstructions (even columns).
- **trav.png**: traversal images.  
- **log.txt**: All losses computed during training.
- **config.txt**: training configurations.

### Help
Important arguments:

```
Generative model:
  --latent_dim          	dimension of the latent variable
  --prior {linscm, nlrscm, gaussian, uniform}
  			               	prior distribution p_z (linear SCM, Nonlinear SCM, or independent ones)
  --labels {smile, age, pend}
                        	name of the underlying structure
                     
Supervised regularizer:
  --sup_coef          		coefficient of the supervised regularizer
  --sup_prop          		proportion of supervised labels
  --sup_type {ce, l2}       type of the supervised loss

Dataset:
  --dataset          		name of the data
  --data_dir          		directory of the dataset
```


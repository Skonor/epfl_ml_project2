# epfl_ml_project2

This reposotory contains code needed to reproduce our experiments for the second project done as a part of EPFL Machine Learning Course.

Code is organaised in the following way. Files code_generation.py contain functions that generate noisy codewards, baseline deterministic maximum likelihood decoder and generation of conjugate (complementary) matrix, that is needed for direct gradient descent setup. File denoisiong_transformer.py contains transformer architecture adapted for decoding task, that includes 3 variations of positional encoding. File datasets.py contains torch implementations of Datasets. 

File train_transformer.py contains code needed to train transformer on the decoding task. 

```
python train_transformer.py
```

This will save a trained model, as well as produce loss plots. 

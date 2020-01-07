## Obtaining the models

Models can be trained using standard functionalities of espnet. The modified
version of the code offered here automatically saves the randomly initialized
model before training under `exp/<exp_name>/results/model.acc.init`.
Alternatively, the models used to generate the results from [1] (random and
pretrained) can be downloaded from [here](www.test.com). The trained model
correspond to the large transformer model with specaug described in
[RESULTS.md] (first model). The random model is the corresponding non trained
version of the same model.

## Extracting activations

Once you have the models, the activations can easily be extracted by running:
```
./extract_activations.sh \
    --decode_config conf/tuning/decode_pytorch_transformer_large.yaml \
    --tag phonemerepr
```

After the script ends, the activation files can be found under the experiment
folder (`exp/train_960_pytorch_phonemerepr`).

## References

[1] Paper
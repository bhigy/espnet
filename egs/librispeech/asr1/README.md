## Obtaining the models

Models can be trained using standard functionalities of espnet. The modified
version of the code offered here automatically saves the randomly initialized
model before training under `exp/<exp_name>/results/model.acc.init`.
Alternatively, the models used to generate the results from [1] (random and
pretrained) can be downloaded from [here](). The trained model correspond to
the large transformer model with specaug described in
[RESULTS.md](https://github.com/bhigy/espnet/blob/phoneme-repr/egs/librispeech/asr1/RESULTS.md#pytorch-large-transformer-with-specaug-4-gpus--large-lm).
The random model is the corresponding non trained version of the same model.

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

[1] Chrupała, Grzegorz, Bertrand Higy, and Afra Alishahi. “Analyzing Analytical Methods: The Case of Phonology in Neural Models of Spoken Language.” In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Seattle, WA, USA: Association for Computational Linguistics, 2020.

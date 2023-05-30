Train model:
```
python bpm_model.py --load CKPT_PATH
```
Model states will be automatically saved to ../data/model_checkpoints
Evaluate model:
```
python bpm_eval.py CKPT_PATH
```
where CKPT_PATH is a path to a saved model state
## Adaptions
The original authors [1] used a Korean sentiment dictionary that assigned each word in a collection one out of five sentiment categories (very negative, negative, neutral, postive, very positive). For a given utterance, they counted the number of categories that occured, resulting in a count vector, e.g. [0, 2, 1, 0, 0] if 2 positive words and 1 neutral word occured. They then normalized this vector by the total number of words, e.g. [0, 0.2, 0.1, 0, 0] if the above utterance contained a total of ten words. In a multitask fashion they let the model predict this vector and propagate the binary cross-entropy loss.

In this implementation we used a [pretrained](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) sentiment classifier to assign each utterance a hard label, i.e. postive, neutral or negative. In a multitask fashion we let the model predict this label and propagate the cross-entropy loss.

## Results
**Table 1:** Classification results in F1 for each class and their weighted F1 for “All” as measured on the Switchboard Corpus Backchannel Dataset [2].
| Model  | No-BC  | Continuer  | Assessment  | All  |
|---|---|---|---|---|
| BPM_ST  | 79.3  | 41.1  | **50.8**  | 62.9  |
| BPM_MT  | **79.8(+0.5)**  | **41.5(+0.4)**  | 50.4(-0.4)  | **63.1(+0.2)**  |

## Citation
[1] Jang, J. Y., Kim, S., Jung, M., Shin, S., & Gweon, G. (2021, November). BPM_MT: Enhanced Backchannel Prediction Model using Multi-Task Learning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 3447-3452).

[2] Ortega, D., Li, C. Y., & Vu, N. T. (2020, May). Oh, Jeez! or uh-huh? A listener-aware Backchannel predictor on ASR transcriptions. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 8064-8068). IEEE.

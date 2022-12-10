# Neural orthographic conversion and normalization for Western Sierra Puebla Nahuatl with special attention to Spanish words.

To run an experiment:
`cd scripts`
`python train.py -m -s` where -m is for running the multitask experiment, and -s means "use supplementary spanish data",

### Intro & Background
This project addresses dealing with varying orthographic norms in written Nahuatl using a sequence-to-sequence neural network model.
Its motivations stem from the creation of the Universal Dependencies treebank for Western Sierra Puebla Nahuatl, a corpus where all of the 
proposed written standards are represented in addition to examples of mixtures of these and other alphabets. If we are working with texts 
that use a specific orthography consistently, converting that text to another of the written standards is relatively trivial using a rule-based
system like a Finite State Transducer. However, this approach is complicated by two phenomena:

1. for projects like corpus aggregation, it isn't always known which orthographic standard is used, and even if one is used it may not be used consistently, and
2. Nahuatl speech and text frequently uses Spanish loan words and/or code-switching. When this happens, the orthographic conventions for these words typically follow the Spanish spelling convention.

About 15% of the words in the UD treebank can be classified as "Spanish", whether that refers to a loan that hasn't undergone significant phonological adaptation or an instance of word-level codeswitching.

### Approach
I attempted to build a neural network model capable of taking input from any latin-alphabet-based Nahuatl orthography (though in practice this just means any orthography represented in our data), and 
convert it to any of the 4 proposed writing "standards" for Nahuatl:

- INALI (proposed by the National Institute of Indigenous Languages in Mexico),
- SEP (proposed and used by the Secretary of Education for Nahuatl materials),
- ILV (orthography used and promoted by the Summer Institute of Linguistics),
- ACK (an orthography based on colonial era practices, developed by three contemporary Nahuatl scholars: Andrews, Campbell, and Karttunen).

Furthermore, I wanted my system to preserve the Spanish orthography for Spanish words, regardless of the desired output orthography (thus, "pues" does not change to "pwes" when converting to INALI orthography).

I used an LSTM-based encoder-decoder architecture with additive attention (Bahdanau et al. 2014). I also evaluated a multi-task training approach, where the encoder output also feeds a language classifier, in 
an attempt to improve the conversion (or rather the lack of conversion) of Spanish words. The idea is that, if the model is also trained to detect Spanish words, then it will do a better job of recognizing when 
it doesn't need to apply an orthographic conversion. Furthermore, I also experimented with adding additional Spanish words to the training data to achieve better performance on Spanish words.

### Data
I used the unique words from the UD Western Sierra Nahuatl corpus, getting conversions into the 4 major orthographic standards mentioned above. I generated the conversions first using an FST, and then manually correcting the conversions.
I also marked each Spanish word. I split the data into train, dev, and test sets, using an approximate 0.8 - 0.1 - 0.1 split. 

For supplementing the data with additional examples of Spanish words, I used the unique words from the Spanish translations in the UD treebank.

Data volume:

| Data set   | Num words     | 
|--------------|-----------|
| train | 16,013      | 
| dev      | 1,881  |
| test      | 1,977  | 
| train_with_extra_spanish      | 22,749  |


### Experiments
I ran four total experiments:
1. Simple seq2seq with attention, on just the Nahuatl UD data (orig)
2. Seq2seq with attention and multitask training on language classification (MTT_orig)
3. Simple seq2seq with attention, trained on Nahuatl UD data AND words from the Spanish translations in the same dataset (xtra_spa)
4. Seq2seq with attention and multitask training on language classification, trained on Nahuatl UD data AND words from the Spanish translations in the same dataset (MTT_xtra_spa)

### Results

| Experiment   | Accuracy     | Accuracy on Spanish words|
|--------------|-----------|------------|
| orig | **0.98**      | 0.88        |
| MTT_orig      | 0.97  | 0.88       |
| xtra_spa      | 0.97  | 0.89      |
| MTT_xtra_spa      | 0.97  |**0.9**      |

### Analysis and Concluding Remarks
Using only the original data, multi-task training didn't have much effect. This could likely be due to the label skew in the data, sicne only 15% of the words are Spanish. When adding the extra Spanish words, the classification-based multi-task training method improved the accuracy on Spanish words by 2 percentage points with only a 1-point drop in overall accuracy comparing to the original experiment (no MTT, no extra data). 

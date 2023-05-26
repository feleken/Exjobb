
Methods
The main task of this thesis was to build a self-supervised pretraining model and a
supervised fine-tuning model. The concept of the models was based on the BERT-
model, mentioned in section 2.6. Thus, the pretraining model used MLM to enable
self-supervised training. Furthermore, they were designed with a backbone and a
head, where the pre-trained backbone was reimplemented in the fine-tuning model.
The general architectures of the models are shown in Figure 3.1, and the methodol-
ogy of designing and optimizing the models will be described in more detail in the
following sections.
Data management and forming experiments were essential parts of the method as
well. The data management was done to prepare the data for the task and to
ensure that defects like e.g. data leakage were avoided. Following, forming relevant
experiments was crucial to determine the performance of the models, as well as to
understand their behaviors.
Figure 3.1: Pre-taining and fine-tuning models. The general architecture of
the pre-training and fine-tuning models, where the backbone from the pertaining
model, containing the upscaling- and encoder block, was reimplemented in the fine-
tuning model.
17
3. Methods
3.1 Data Management
The data used in this project consisted of processed data from Smart Eye’s driver
monitoring software. The models were therefore not fed with raw images or videos,
but with numerical data that had been extracted from the videos of the different
drivers. In this thesis, only features related to the eye blink of a driver were used,
where up to nine features were selected in total. In addition to the original nine
features, there are a number of additional features that were derived from the original
ones, e.g. calibration features, which can be used as a tool to standardize every
driver’s features.
The data was processed in several ways to fit the method. The original training data
consisted of one data frame for each recording, where every row represented a blink
and the columns represented the different features of the blink. All the data frames
in the training data were then stacked, i.e. the first N rows in the stacked data frame
were blinks from the first recording, then from row N+1 were blink from the second
recording, and so on. The stacked data frame was then used to create time series
data sets, using PyTorch Forecasting TimeSeriesDataSet [18], i.e. each sample is a
subsequence from each recording. It was however taken into consideration to not let
the subsequences overlap two different recordings, as they were stacked. Lastly, the
time series data sets were used to create a data loader for the training set. This was
done by a method in the timeSeriesDataSet class which creates batches of different
samples to be used for training the model. This entire process was repeated for both
the validation and test set as well.
For the purpose of the project, the available data had to be split into two more
data sets, one for pre-training and one for fine-tuning. When forming these two
sets, it was important that there were no overlapping drivers in the different sets in
order to avoid data leakage. Furthermore, the fine-tuning set should only include
high-quality data, where the labels are known to be good, and the pre-training set
included any data with blinks, meaning that labeled data of lower quality was also
included.
3.2 Self-supervised Pre-training Model
The pre-trained model’s architecture was built from scratch and was inspired by the
BERT-model described in section 2.6. For example, MLM was implemented to be
able to benefit from self-supervised learning, and the encoder block in the backbone
was designed in a very similar way as the one mentioned in section 2.5.
Yet, designing the pre-training model was a wide-ranging task, partly due to the
differences between the nature of the data fed to the original BERT-model and the
data in this thesis, and partly due to the many different ways the remaining parts of
the model could be built, and the number of parameters to tune. So, in the following
sections, the methodology to translate MLM to the data in this thesis, and to design
the pre-training model’s all components in the best way possible, will be explained
in further detail.
18
3. Methods
3.2.1 MLM on Blink-features
As the original architecture of MLM is based on text data, it had to be customized
to fit the data in this thesis, which is vectors containing continuous values. In the
MLM there are three different actions that can be applied to the input data. One is
to let it remain unchanged, one is to replace it with an arbitrary mask index, and the
last one is to replace it with a random value. To let the data remain unchanged was
thus the most trivial operation to translate since it could be done in the exact same
way with different types of data. Furthermore, the operation where data is replaced
with a mask index was rather straightforward as well. The operation simply replaces
all the elements in the vector with the chosen mask index, which in this thesis was
set to −1.
The operation when the data is replaced with a random value was however not as
trivial to translate, due to the fundamentally different nature of the data in the two
cases. When the operation is made on text data, the random value is a token chosen
from a vocabulary available in the task, consisting of a finite number of tokens. Since
the data in this thesis consists of continuous data, on the other hand, the random
value can, theoretically, be set to an infinite choice of values. As the purpose of
the action is to add noise as a regularization method, the decision was made not to
replace values with totally random ones, but to tailor the operation to this specific
task. Thus, when a vector of continuous values was chosen to be randomized, each
element was multiplicated with a noise factor, where the noise factor was chosen
randomly but still within a chosen interval.
Figure 3.2: Randomizing vectors in MLM. Example of the operation where a
vector of continuous values is randomized by being multiplicated with noise factors.
In summary, the MLM takes input data with shape (B, L, N ), where B is the batch
size, L is the sequence length, and N is the number of features. In each batch, 15%
of the blinks in the sequence, i.e. rows, are chosen randomly to be masked with one
of the three different masking options. The first option, to replace all features in the
row with the masking index −1 is applied to 80% of the chosen rows. The second
and third options, to randomize the features in a row with a noise constant and to
leave a row unchanged, are applied to 10% each. The masked data is then returned,
together with indices of which rows have been masked to be able to extract them
19
3. Methods
for the prediction that will be done further on in the method. An example of how
data can look before and after it’s been processed in the MLM is shown in Figure
3.3.
Figure 3.3: Data before and after MLM. Example of how data can look before
and after being processed by the MLM block. The rows in darker blue are the ones
that were handled, and were either replaced with the mask index (-1), randomized
with noise, or kept the same.
3.2.2 Upscaling Block
As mentioned in section 2.5.2, there are multiple different ways of performing feature
upscaling depending on the task at hand. In this thesis, four different alternatives
were evaluated, which were no upscaling at all, linear upscaling, and two different
attention-based upscaling methods.
The first method, to not use any upscaling at all, is to simply add the positional
embedding right onto the input data. This method naturally leads to a less complex
network with fewer trainable parameters, and the embedding size being equal to the
input size throughout the entire model. The second method, linear upscaling, is a
bit more complex as it increases the number of trainable parameters and expands
the embedded size.
The attention-based upscaling methods, however, are more complex in their nature.
Both of them pass the input data through an encoder block with self-attention, to
caption the relations within the data. The first method then upscales the encoded
data with a linear layer, whilst the other one instead has a skip-connection of the
input data which is upscaled with a linear layer, and concatenated with the encoded
data.
All four of the methods were implemented and compared against each other to find
the one that yields the best performance. How this was done will be explained in
the next section.
20
3. Methods
3.2.3 Hyperparameter Sweep
As mentioned in section 2.8, it can be crucial to perform hyperparameter sweeps
to find the optimal combination of settings of the hyperparameters. The process to
find the optimized combination of hyperparameters in this thesis was made using
the Python plugin Optuna, which is an automatic hyperparameter optimization
software framework. By using Optuna, the sweep could be done in a more time-
efficient and accurate way. It does however have limitations, e.g. GPU performance
and memory, which restricted the scope of the search space.
The chosen search space is presented in Table 3.1, as well as the best value of each
setting that was found. The sweep ran for 50 trials, where the first 25 of them were
start-up runs, i.e. runs where the hyperparameters were chosen randomly from the
search space and previous results weren’t considered.
Table 3.1: Hyperparameter sweep of the pre-training model. The different
hyperparameters in the pre-training model that were explored in the sweep, with
their respective ranges or sets of values to explore, and finally the best combination
of settings found.
Hyperparameter Explored range or set Best
Upscaling
[None, Linear,
Attention+Linear,
Attenion+Concatenation]
Attention+Linear
Embedding size [32, 64, 128, 256, 512] 512
Encoder blocks [2, 4, 8] 4
Attention-heads in encoder [2, 4, 8] 4
Dropout (Backbone) [0.0, 0.01, 0.02, ..., 0.2] 0.01
Hidden sizes (Head) [[64, 64], [32, 32], [16, 16],
[8, 8], [64], [32], [16], [8]] [64]
Dropout (Head) [0.0, 0.01, 0.02, ..., 0.2] 0.02
Batch normalization (Head) [True, False] True
Warmup epochs [0, 1, 2, 3, 4, 5] 4
Learning rate [0.00001, 0.00002, ..., 0.001] 0.00062
Accumulate gradient batches [1, 2, 4, 8, 16, 32] 1
Decay factor [0.85, 0.90, 0.95, 1.0] 0.9
Weight decay [0.0, 0.0001, 0.0002, ..., 0.01] 0.0095
Gradient clip value [1.0, 2.0, 4.0, 8.0] 1.0
Furthermore, the performed sweep also provided results on how important the differ-
ent parameters were for the performance, which is shown in Figure 3.4. The results
show that the choice of upscaling method as well as the embedding size were the
two parameters that had the biggest impact on the performance. Furthermore, the
values of them that yielded the best result, i.e. attention-based upscaling and em-
bedding size of 512, are, together with attention-based upscaling with concatenation,
the most complex settings in the explored ranges.
21
3. Methods
Figure 3.4: Importance of hyperparameters in pre-training model. Bar
chart of the importance of each hyperparameter included in the sweep session related
to the pre-training model.
Finally, when the sweep was done, the final architecture of the pre-training model
was formed, which will be described in greater detail in the following section.
3.2.4 Architecture of Pre-training Model
Based on the hyperparameter sweep, the final architecture of the pre-training model
could be determined. As previously mentioned, the architecture was influenced both
by the BERT model, and the encoder in [19]. From these architectures, some own
implementations, like a customized MLM block and different upscaling methods,
were made. To find the best combination of settings, a hyperparameter sweep was
done. Based on the results from the sweep, the final pre-training model was designed.
In Figure 3.5, the architecture of the backbone is visualized. The chosen settings in
the backbone’s layers are the ones that were found in the sweep, presented in Table
3.1 in section 3.2.3. In the pre-training model, the initial input, which consists of
batches with time series of blink-features, is fed into the MLM block described in
section 3.2.1. The output from the MLM block, which is of the same shape but with
some rows masked, is then fed into the backbone. The data is then passed through
the backbone’s upscaling block, and then, with positional embedding added to it,
passed through four encoder blocks. Worth mentioning is that the attention layers
are adjusted to not consider masked and padded rows. When the data has been
processed through the backbone, it’s returned as an output of the same shape as
the initial input shape.
22
3. Methods
Figure 3.5: Backbone’s architechture. Architecture of the backbone, with an
upscaling block with self-attention and linear upscaling, positional embedding, and
four identical encoder blocks.
When data has been processed by the MLM block and the backbone, it is passed
on to the pre-training model’s head, which consists of a regression block, shown in
Figure 3.6. The hyperparameters in the head’s layers are also chosen based on the
sweep. Finally, the head’s output will also be of the same shape as the initial input.
However, only the rows that were masked in the MLM block will be considered when
the loss is calculated.
23
3. Methods
Figure 3.6: Pre-training model’s head’s architecture. Architecture of the
head in the pre-training model, with an output with the same shape as the input in
the head. The rows that were masked in MLM, in darker blue, are then filtered out
from the output and used to calculate the loss.
3.2.5 Baseline Performance
Determining how well the performance of a model is can be a non-trivial task, as a
measured performance often is without context. So, in order to get a sense of how
well the performance of the models in this thesis was, some baseline performances
were used for comparison. The baseline performances were simple solutions to the
objectives, i.e. could be achieved without any advanced machine learning methods.
The objective of the pre-training model was to predict the rows that were masked
during MLM and to diagnose it’s performance, two different baselines were chosen.
The first one was to calculate the mean value of each feature in every sample, and
the other one was to set the prediction to be equal to the next row in the sample.
Both baselines were produced on the unmasked data, i.e. the calculated mean
values were based on the entire sample, and the next row was always available. In
a masked sample, the true mean can’t be calculated due to the missing data, and
since the masking is random, two adjacent rows could be masked which makes it
impossible to use the next row as a prediction. Thus, the baselines did somehow have
some advantages compared to the pre-training model but were altogether simpler
solutions.
3.3 Supervised Fine-tuning model
The fine-tuning model was designed with the same general structure as the pre-
trained model, i.e. a backbone in the form of an encoder (see Figure 3.5) followed
by an output head. Similar to the pre-trained model there were a few required steps
24
3. Methods
to finish before being able to start training. The process included performing an
additional hyperparameter sweep for the output head, exploring different options for
transferring knowledge from pre-trained encoders, as well as evaluating the different
encoder setups. The following sections will explain these steps in greater detail.
3.3.1 Hyperparameter Sweep
A similar hyperparameter sweep as the one in section 3.2.3 was set up for the fine-
tuning model using the Optuna plugin. The difference though, was that it only
considered parameters for the output head and some general setup variables since
the specification of the encoder was given in the earlier hyperparameter sweep.
The search space for the selected parameters was the same as before, and the re-
sulting parameters are presented in Table 3.2. Due to a smaller search space and a
shorter compilation time per trial, the sweep ran for approximately 100 trials, but
only the first 15 were start-up runs used for exploring randomly.
Table 3.2: Hyperparameter sweep of the fine-tuning model. The different
hyperparameters in the fine-tuning model that were explored in the sweep, with
their respective ranges or sets of values to explore, and finally the best combination
of settings found.
Hyperparameter Explored range or set Best
Hidden sizes [[64, 64], [32, 32], [16, 16],
[8, 8], [64], [32], [16], [8]] [16]
Dropout [0.0, 0.01, 0.02, ..., 0.2] 0.16
Batch normalization [True, False] True
Accumulate gradient batches [1, 2, 4, 8, 16, 32] 32
Gradient clip value [1.0, 2.0, 4.0, 8.0] 1.0
As in the previous sweep, the result also contained information on the most im-
portant parameters for the evaluated model, which is presented in Figure 3.7. For
the fine-tuning model, the most important hyperparameter was the choice of using
batch normalization or not, followed by the hidden sizes. Moreover, as the figure
shows, the best choices for these parameters were to use batch normalization and
have a hidden size of 16.
25
3. Methods
Figure 3.7: Importance of hyperparameters in the fine-tuning model. Bar
chart of the importance of each hyperparameter included in the sweep session related
to the fine-tuning model.
3.3.2 Unfreezing Method
As mentioned in section 2.4 an important part of transferring knowledge from a
pre-trained model is to gradually freeze and unfreeze layers. During fine-tuning, the
encoder block from the pre-trained model was initialized with all its layers frozen
for the first three epochs. After the third epoch, the process of gradually unfreezing
layers was executed in two steps. First, the top encoder block layer was unfrozen at
epoch four, followed by the next encoder layer at epoch 6. This resulted in a final
model where the first layers were kept frozen during the entire training, and the
two topmost encoder blocks got fine-tuned to the new dataset and task. However,
multiple different settings for gradually unfreezing layers were evaluated to find the
optimal version, and the differences can be seen in Figure 4.4.
3.3.3 Architecture of Fine-tuning Model
After finishing the hyperparameters sweep and exploring different methods of grad-
ually unfreezing the pre-trained layers, the design of the fine-tuning model was
complete. As mentioned in section 3.3 the architecture consisted of a backbone and
an output head, where the backbone was either a pre-trained encoder or an encoder
trained from scratch. Both of which had the architecture seen in Figure 3.5 with the
chosen parameters presented in Table 3.1. Although, the latter one was only trained
on the fine-tuning data set throughout the entire model, without MLM since that
is a pre-training technique.
After the data had been passed through the backbone it was processed in a way such
that only the last, non-padded row was kept for each blink sequence. "The reason for
this was that" the fine-tuning output head, seen in Figure 3.8, predicted drowsiness
in the form of one KSS-value for each sequence and therefore only needed one input
sample for each sequence. Hence, the output head was designed as a regression head
which transformed each feature vector into a KSS value on a scale of 1-9.
26
3. Methods
Figure 3.8: Fine-tuning model’s head’s architecture. Architecture of the
head in the fine-tuning model, with a single KSS-value for each sample in a batch
as output.
The training and evaluation of the fine-tuning model were as mentioned done with
both a pre-trained encoder and an encoder that was trained from scratch. Both of
these cases were optimized with the 4-level drowsiness KPI mentioned in section 2.7,
and evaluated on multiple different metrics to get a broad view of the performance
of the models.
3.4 Experiments
To be able to answer the research question it was required to perform some different
experiments on the models. Experiments were done on the number of features and
the amount of data used in the training process, for both the pre-trained and fine-
tuned model. On the pre-trained model, three different experiments were conducted.
One with all the available data and all 18 features, one with all the data but only
the original features and not the calibration, and the last one was built upon the
result of the first two. I.e. the feature setup with the best result was used in another
experiment with only half the data to see the impact of the amount of data. Lastly,
three additional experiments were performed on the fine-tuning model. One with
the pre-trained encoder on all data, one with half the data, and one with an encoder
trained from scratch i.e. no pre-training.

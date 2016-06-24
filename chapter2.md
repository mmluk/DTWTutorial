---
title       : Dynamic Time Warping
description : Dynamic time warping is a useful similarity score that allows comparisons of unequal length time-series, with different phases, periods, and speeds.
attachments :
  slides_link : http://sflscientific.com/presentations-and-conference-talks/

--- type:MultipleChoiceExercise lang:python xp:50 skills:1 key:5277a8088f
## Introduction

Typical time-series techniques usually apply algorithms, be it SVMs, logistic regressions, decision trees etc,  after transforming away temporal dynamics through feature engineering. By creating features, we often remove any underlying temporal information, resulting in a loss of predictive power.

Dynamic time warping (DTW) is a useful distance-like/similarity score that allows comparisons of two time-series sequences with varying lengths and speeds. Simple examples include detection of people 'walking' via wearable devices, arrhythmia in ECG, and speech recognition. 

This measure distinguishes the underlying pattern rather than looking for an exact match in the raw time-series. As its name suggestions, the usual Euclidean distance in problems is replaced with a dynamically adjusted score. DTW thus allows us to retain the temporal dynamics by directly modeling the time-series. 

Much of the following material is taken from our blog and case studies from our website: http://www.sflscientific.com

## DTW as a distance

Figure 1 shows what happens when we compare two time-series, symbolised as a red wave and a blue wave. The top image shows Euclidean matching, which is 1-1. The similarity score between these waves measured by a Euclidean metric would be poor, even though the rough shape of the waves (i.e. the peaks, troughs and plateaus) are similar.

The bottom image shows a similarity score that would be given by Dynamic Time Warping. You can see that the rough shapes of the waves would match up, giving a high similarity score. Notice that DTW is not exactly a distance metric as the triangle inequality does not hold.


*** =pre_exercise_code
```{r}
# The pre exercise code runs code to initialize the user's workspace.
# You can use it to load packages, initialize datasets and draw a plot in the viewer

from IPython.display import Image
import matplotlib.pyplot as plt
%matplotlib inline
%pylab inline
Image('images/dtw_dummy.jpeg')
```


Have a look at the plot that showed up in the viewer. Which type of measure would give a greater similarity between red and blue curves, when you use the black vertical lines to compare points?

*** =instructions
- Euclidean matching
- DTW matching

*** =hint
Have a look at the plot. Are points in the curves connected by the black lines more or less similar.


*** =sct
```{r}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

msg_bad = "That is not correct!"
msg_success = "Exactly!"
test_mc(2, [msg_bad, msg_success])
```

In general, events that have similar shapes but different magnitudes, lengths and especially phases can prevent a machine from correctly identifying sequences as similar events using traditional distance metrics. DTW allows us to get around this issue by asynchronously mapping the curves together.


*** =pre_exercise_code
```{r}
Image('images/path_differences.png')
```
The figure above left shows the typical Euclidean matching between two waves.  Starting in the bottom left, the first instance in the sequence of the time-series A and B are compared to each other. Then the second instance is compared to the second and so on until the end of one of the shorter sequences.

For DTW, the figure above right, represents a walk over the optimal path. The optimal path is determined by finding the maximum similarity score between the two time-series.


*** =pre_exercise_code
```{r}
Image('images/allpaths.png')
```
To find the optimal path, DTW checks all possible paths (subject to certain constraints) from the bottom left to the top right, computing the equivalent of a similarity score between the waves for each. The one with the largest similarity is kept.


--- type:NormalExercise lang:python xp:100 skills:1 key:10223ad896
## A Simple Example

Let's start with a naive speech recognition example of DTW to show how the algorithm works, and then we will suggest a more complicated version of the analysis that can be found on our website.

A dataset of file labels, `labels`, and data `data`,  is available in the workspace.

*** =pre_exercise_code
```{python}
# import scipy libraries to load data
import scipy.io.wavfile
import scipy.signal as sig

with open('data/sounds/wavToTag.txt') as f:
    labels = np.array([l.replace('\n', '') for l in f.readlines()])

data = []
for i in range():
  data.append(scipy.io.wavfile.read('data/sounds/{}.wav'.format(i))[1])

import numpy as np
```

Both `labels` and `data` are stored in numpy arrays and can be accessed as a standard array. Let's have a look at what the raw data looks like.

*** =instructions
- What labels to the files 0 and 8 have? 
- Import matplotlib.pyplot as `plt`
- Use `plt.scatter()` to plot `data[0]` and `data[8]` onto the same image. You should use the first positional argument, and the `c` keyword.
- Show the plot using `plt.show()`.

*** =hint
- You don't have to program anything for the first instruction, just take a look at the first line of code.
- Use `import ___ as ___` to import `matplotlib.pyplot` as `plt`.
- Use `plt.scatter(___, ___, c = ___)` for the third instruction.
- You'll always have to type in `plt.show()` to show the plot you created.


*** =sample_code
```{python}
# Show the labels for file 0 and 8.
plot('label for file 0 is:,labels[0])


# Import matplotlib.pyplot


# Make a scatter plot: with data of files 0 and 8 and set c to ints


# Show the plot

```

*** =solution
```{python}
# Get integer values for genres
print('label for file 0:',labels[0])
print('label for file 8:',labels[8])

# Import matplotlib.pyplot
import matplotlib.pyplot as plt


# plot the raw audio data
plt.plot(data[0], label='Sample '+str(id1))
plt.plot(data[8],alpha=0.2, label='Sample '+str(id2))
plt.title('Raw data for two speakers',size=20)
plt.ylabel('Amplitude')
plt.xlabel('Measurement')
plt.legend()


# Show the plot
plt.show()
```

*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki


test_object("data",
            undefined_msg = "Don't remove the definition of the predefined `ints` object.",
            incorrect_msg = "Don't change the definition of the predefined `ints` object.")

test_import("matplotlib.pyplot", same_as = True)

test_function("matplotlib.pyplot.plot",
              incorrect_msg = "You didn't use `plt.scatter()` correctly, have another look at the instructions.")

test_function("matplotlib.pyplot.show")

success_msg("Great work!")
```


Let's also check the lengths of the file:
```{python}
print('File Lengths:',len(data[0]),',',len(data[8]))
```

Notice the file lengths are significantly different. Comparing directly two uneven length vectors is already quite unnatural in most situations when using standard distance metrics. 

DTW gets around these problems since the underlying algorithm and distance metrics don't really care about the lengths of the file.

In our data set we have the following different labels:
```{python}
print('unique labels:',set(labels))
```


For computational ease, we will downsample all the data to 1000 Hz, from 44kHz and also normalise the data.
```{python}
# Downsample the data from rate [/s] to new_rate [/s] and normalise (mean and variance) it by hand
rate = 44100
new_rate = 1000
data_sampled = sig.decimate(data[0], int(rate/new_rate), axis=0, ftype='fir')
data_sampled -= np.nanmean(data_sampled)
data_sampled /= np.nanstd(data_sampled)

data_sampled2 = sig.decimate(data[8], int(rate/new_rate), axis=0, ftype='fir')
data_sampled2 -= np.nanmean(data_sampled2)
data_sampled2 /= np.nanstd(data_sampled2)
```

We plot the normalised data for the same samples:
```{python}
plt.plot(data_sampled, label='Sample '+str(id1))
plt.plot(data_sampled2, alpha=0.2, label='Sample '+str(id2))
plt.title('Normalised Data for two speakers',size=20)
plt.ylabel('Amplitude')
plt.xlabel('Measurement')
plt.legend()
```

For convenience, let's put the normalisation into a single function
```{python}
def normalise_data(data, original_rate, final_rate):
  import scipy.signal as sig
   
  data_sampled = sig.decimate(data, int(rate/new_rate), axis=0, ftype='fir')
  data_sampled -= np.nanmean(data_sampled)
  data_sampled /= np.nanstd(data_sampled)

  return data_sampled
```

Let's reshape the data for input into dtw function
```{python}
x = data_sampled.reshape(-1,1)
y = data_sampled2.reshape(-1,1)
```

Let's compute our first DTW distance between these two audio clips:
```{python}
# compute the norm (x-y)
dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
print('Minimum distance found:', dist)
```

Notice that we need to specify a norm. The norm is used to compare the elements of each step, with the sum of these the total cost of the path. We can plot the optimal path that the algorithm using this particular norm as follows:

```{python}
# plot the path
plt.imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot(path[0], path[1], 'w')
xlim((-0.5, acc.shape[0]-0.5))
ylim((-0.5, acc.shape[1]-0.5))
xlabel('Sample '+str(id1))
ylabel('Sample '+str(id2))
plt.title('np.linalg.norm(x-y)')
```

You can also specify your own norm used to determine the cost measure by the DTW as follows:
```{python}
def my_custom_norm(x, y):
    return (x - y)*(x - y)

dist, cost, acc, path = dtw(x, y, dist=my_custom_norm)
print('Minimum distance found:', dist)

# with path
plt.imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot(path[0], path[1], 'w')
xlim((-0.5, acc.shape[0]-0.5))
ylim((-0.5, acc.shape[1]-0.5))
plt.title('custom norm')
```

You can obviously play with the variety of norm you would like, some will be more suitable for different cases as with all normalisation methods and represent a good paramter to tune.

For simplicity, let's just use the built-in norm from numpy, which gives good results (determined by the minimum distance found). 
```{python}
from numpy.linalg import norm
dist, cost, acc, path = dtw(x, y, dist=norm)

imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plot(path[0], path[1], 'w')
xlim((-0.5, acc.shape[0]-0.5))
ylim((-0.5, acc.shape[1]-0.5))
plt.title('np.linalg default norm')
print('Minimum distance found:', dist)
```

Lets apply this to all files in the dataset. For this naive example, let's find one file of each word type and use this as a template training example. 

Obviously this is not ideal, in accuracy or complexity but there are use cases where we know some underlying true 'template' that we want to dig out of some time-series data. 

We will revisit a better algorithm afterwards.

## Naive DTW Classifier - using only minimum distance to training example
```{python}
template_labels = {}
template_data = {}
training_indices = []
new_rate = 200
for l in set(labels):
    first_index = (list(labels).index(str(l)))
    rate,data = scipy.io.wavfile.read('data/sounds/{}.wav'.format(first_index))
    data_sampled = normalise_data(data,rate,new_rate)
    
    template_data[l] = data_sampled
    template_labels[l] = first_index
    training_indices.append(first_index)
```

We will use one example of each label as a traning set.
```{python}
print('The training indices that we will use are:',training_indices)
```
We also relabel our labels as truth, for emphasis:
```{python}
true_label = labels
```

We now loop over all files in our dataset and compare it with each template. The comparison that yields the smallest distance (i.e. the highest similarity) will be used as the label. This is basically a naive k-Nearest Neigbhour algorithm with only one of each label in the training set. 

**THIS WILL TAKE A WHILE!!!**
```{python}
pred_label = []
pred_score = []
print('total files:',len(labels))
for f in range(0,len(labels)):
    if not (f % 25):
        print('working on file',f, 'true label = ',true_label[f])
    test_rate,test_data  = scipy.io.wavfile.read('data/sounds/{}.wav'.format(f))
    
    # down sample and normalise
    data_sampled = sig.decimate(test_data, int(test_rate/new_rate), axis=0, ftype='fir')
    data_sampled -= np.nanmean(data_sampled)
    data_sampled /= np.nanstd(data_sampled)

    # initialise some variables
    min_dist = np.inf
    score_list = []
    for template in template_data:
        # compute the distance to each template one at a time
        dist, _, _, _ = dtw(data_sampled.reshape(-1,1), template_data[str(template)].reshape(-1,1), dist=lambda x, y: np.linalg.norm(x - y, ord=1))
        if dist < min_dist:
            # save the current best template details
            min_dist = dist
            pred = template
        # save the distance score to a list
        score_list.append(dist)
    
    if not (f % 25):
        print('Completed file',f,'closest match',pred)
    
    # save the predicted labels
    pred_label.append(pred)
    
    # save the list of scores
    pred_score.append(score_list)
  ```
  
  
Let's now see how this does in terms of accuracy on the test set. 
```{python}

# define a plotting script for the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Greens):
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=90)
    plt.yticks(tick_marks, label_list)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
```


The plotting script requires a list of ints rather than strs so we convert the list of str to list of ints:
```{python}
# create a dict of str names to int names
label_dict = {}
label_list = []
i=0
for x in set(true_label):
    label_dict[x] = i
    label_list.append(x)
    i+=1

# int classes for the true and predicted. 
true_int = [label_dict[l] for l in true_label]
pred_int = [label_dict[l] for l in pred_label]

# int classes for true and predicted, just the test set.
test_pred_int = [v for i, v in enumerate(pred_int) if i not in training_indices]
test_true_int = [v for i, v in enumerate(true_int) if i not in training_indices]

```


```{python}
cm = confusion_matrix(test_pred_int,test_true_int)
print(classification_report(test_pred_int, test_true_int, target_names=[l for l in label_list]))
    
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized)
```


Notice that this is TERRIBLE! The main reason is that we are using 1 known training example as our template. How would we improve this? If we do something more intelligent with a full kNN as the underlying algorithm and DTW as the distance metric. We do exactly that in the full analysis: 
 slides_link : http://sflscientific.com/presentations-and-conference-talks/

ULtimately, using a k-Nearest Neighbour vote. The methodology of comparing time-sequences is identical to the above, except we use a k-Nearest Nieghbour algorithm:

```{python}
Image('images/dtw_knn_schematic.png',width=780)
```

With the exat same methodology, we find the following confusion matrix:
```{python}
Image('images\final_confmatrix.jpg')
```
which is definitely very reasonable for such a simiple algorithm. 

# Final Thoughts

Just with this minimal setup, we have already achieved very high accuracy without the need for feature engineering.  These methods can be applied in conjunction with any distance-based algorithms and have shown great success in time-series analyses in healthcare, sports, finance and more. 

Notice that whilst the methods we have shown are quite slow, the process above with finding neighbours lends itself nicely to distributed computation, with the usual caveats that come with kNN of memory allocation for the training data. Further, faster implementations of the DTW computation (in particular check out UCR's DTW suite and Python's fast DTW library, which runs an approximation of the DTW).

Finally, we can also attempt to ensemble these techniques with more-standard feature-generation methods to achieve even better results.

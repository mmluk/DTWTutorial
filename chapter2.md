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

Both `labels` and `data` are stored in numpy arrays and can be accessed as a standard array.

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

# Make a scatter plot: runtime on  x-axis, rating on y-axis and set c to ints
plt.plot(data[0], c=ints)
plt.plot(data[8], c=ints)

# Show the plot
plt.show()
```

*** =sct
```{python}
# SCT written with pythonwhat: https://github.com/datacamp/pythonwhat/wiki

test_function("numpy.unique",
              not_called_msg = "Don't remove the call of `np.unique` to define `ints`.",
              incorrect_msg = "Don't change the call of `np.unique` to define `ints`.")

test_object("ints",
            undefined_msg = "Don't remove the definition of the predefined `ints` object.",
            incorrect_msg = "Don't change the definition of the predefined `ints` object.")

test_import("matplotlib.pyplot", same_as = True)

test_function("matplotlib.pyplot.scatter",
              incorrect_msg = "You didn't use `plt.scatter()` correctly, have another look at the instructions.")

test_function("matplotlib.pyplot.show")

success_msg("Great work!")
```

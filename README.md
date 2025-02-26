# Results:

The Naïve Baye’s assumption of independence does not appear to hold true to the attributes
of this dataset. The individual features of this dataset are word frequencies (and symbols),
many of which frequently co-occur in spam emails. This likely contributed to the
poor performance of Naïve Baye’s Classification on this dataset.

Another possibility for poor performance is that the dataset was roughly a 40–60% split
between spam–not spam. There could be possible bias towards classifying emails as not
spam as they take the majority of the dataset.

Accuracy shows how often the model is right, Precision shows how often positive predictions
are correct, and Recall shows whether an ML model can find all objects of the target class.
The higher recall (67.61%) suggests that the model is better at detecting spam, but it has
a high false positive rate when related to the low precision (39.39%). Additionally, the low
accuracy (46.26%) suggests that the model is performing poorly overall.

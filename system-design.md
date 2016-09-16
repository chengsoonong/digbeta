# Thoughts about the design of active learning APIs

Abstract away files, etc and consider only a table (think SQL or CSV with header).

Let's use the working title of ```suggest``` for the active learning query generator.

The classifier passes a table that contains:
1. A column identifying the row (ML: example, DB: primary key)
2. A list of columns containing predictions (assume to be scalar real numbers)
```
suggest db.table id_col pred_col1 [pred_col2, ...]
```

```db.table``` may contain other columns (which should not be peeked at).
```id_col``` is a string containing the column name of the identifier
```pred_colk``` is a string containing the prediction of predictor *k*

The recommender runs whatever it needs to run, and returns a value from ```id_col```.

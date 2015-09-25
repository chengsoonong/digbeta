* ```filtering-bigbox.py```   
Python3 scripts to filter out photos outside the big bounding box from YFCC100M dataset  
```./filtering-bigbox.py yfcc100m_dataset-0```

* ```generate-tables.py```  
Python3 scripts to generate trajectories (i.e. the two tables) with respect to the small bounding box  
```./generate-tables.py Melbourne-bigbox.csv```   
where the input file is result after filtering

* ```visualise.py```
Python3 scripts to generate (a number of) KML files to visualise trajectories  
```./visualise.py  Melb-table1.csv  Melb-table2.csv``` 
where the input files are the output of ```generate-tables.py```

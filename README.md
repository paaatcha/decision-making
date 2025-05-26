# Decision-making algorithms 

Welcome to the decision-making repository :shipit:

This repository contains a complete implementation in Python of the following decision-making algorithms:
- **TOPSIS:** Technique for Order Preference by Similarity to Ideal Solution
- **TODIM:** an acronym in Portuguese for Interative Multi-criteria Decision Making

In addition, it also contains the framework **A-TOPSIS**, which is quite useful to rank a set of algorithms based on their
performance for group of benchmarks.

For more information about the methods, please, refer to the [references section](#references).

If you find any bug or just want to contribute with a new feature or new algorithm, feel free to open a new issue an
pull request to the repository.

Also, if this code was useful for you, consider leaving a star :star: and checking the [citation section](#citation) :books:


# How to use this package

First of all, you must install the dependencies by running the following command:
```commandline
pip install requirements.txt 
```

The package is organized as follows:
- `examples`: this folder contains an example for each algorithm. Use it to understand how to use them :teacher:
- `src`: it contains the Python package with the implementation for each method :octocat:
- `test`: as the name suggest, it contains the unit test for each method. To run it, you must run the `pytest` command
inside the folder :test_tube:

All methods are documented using the docstring format. However, to understand how the algorithms work, you must
refer to their paper linked in the [references section](#references).

# Running an automatic A-TOPSIS test

Now, you can export your benchmarck's results from `deep`

Add your aggregated results inside the folder 'dataset', change the configuration of the a-topsis in the script 'scripts/atopsis_from_file.py' and then run the command: `python3 scripts/atopsis_from_file.py`


## References
All implementations follow the standard approach described in the paper. However, for TODIM algorithm, we included the
updated decribed by Lourenzutti and Khroling (2013), since the original algorithm seems to be broken for some cases.  


#### For more information about TODIM, please, refer to:

- L.F.A.M. Gomes, M.M.P.P. Lima. TODIM: Basics and application to multicriteria ranking of projects with environmental 
impacts Foundations of Computing and Decision Sciences, 16 (4) (1992), pp. 113-127


- Lourenzutti, R. and Khroling, R. A study of TODIM in a intuitionistic fuzzy and random environment,
Expert Systems with Applications, Expert Systems with Applications 40, (2013), pp. 6459-6468

#### For more information about TOPSIS, please, refer to:

- C.L. Hwang & K.P. Yoon, Multiple Attributes Decision Making Methods and Applications, Springer-Verlag, Berlin, 1981.


#### For more information about A-TOPSIS, please, refer to:
- Krohling, R. A., and Pacheco, A.G.C. A-TOPSIS - an approach based on TOPSIS for ranking evolutionary algorithms. 
Procedia Computer Science 55 (2015): 308-317.


- Pacheco, A.G.C. and Krohling, R.A. "Ranking of Classification Algorithms in Terms of 
  Mean-Standard Deviation Using A-TOPSIS". Annals of Data Science (2016), pp.1-18.


## Citation
If this package was useful for you, consider citing the papers I wrote when I was developing my research in this field:

```
@article{krohling2015topsis,
  title={A-TOPSIS--an approach based on TOPSIS for ranking evolutionary algorithms},
  author={Krohling, Renato A and Pacheco, Andr{\'e} GC},
  journal={Procedia Computer Science},
  volume={55},
  pages={308--317},
  year={2015},
  publisher={Elsevier}
}
```

```
@article{pacheco2018ranking,
  title={Ranking of classification algorithms in terms of mean--standard deviation using A-TOPSIS},
  author={Pacheco, Andr{\'e} GC and Krohling, Renato A},
  journal={Annals of Data Science},
  volume={5},
  number={1},
  pages={93--110},
  year={2018},
  publisher={Springer}
}
```

```
@article{lourenzutti2013study,
  title={A study of TODIM in a intuitionistic fuzzy and random environment},
  author={Lourenzutti, Rodolfo and Krohling, Renato A},
  journal={Expert Systems with Applications},
  volume={40},
  number={16},
  pages={6459--6468},
  year={2013},
  publisher={Elsevier}
}
```


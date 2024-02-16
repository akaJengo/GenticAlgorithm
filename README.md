
# Genetic Algorithm

This project uses the Python DEAP library to classify malignant or benign tumors using the Wisconsin breast cancer data set. 


## Installation

Pip packages required

```bash
  DEAP
  Numpy
  Pandas
```
    
## API Reference

#### Running program
These paramters can be changed prior to running the Classification algorithm or the Regression algorithm. 
```http
  python3 params.py 
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `Seed` | `integer` | **Required**. The seed for RNG |
| `maxDepth` | `integer` | **Required**. Depth of trees |
| `tournamentSize` | `integer` | **Required**. Tournament size |
| `popSize` | `integer` | **Required**. Size of population |
| `terminals` | `integer` | **Required**. Running instances of the program |
| `crossoverRate` | `integer` | **Required**. Genome crossover rate|
| `mutateRate` | `integer` | **Required**. Genome mutation rate |
| `numGenerations` | `integer` | **Required**. Number of generations|
| `file` | `string` | **Required**. File path for output |




## Deployment

To deploy this project run

```bash
  python3 Classification.py 
```


## Papers
### Testing Paramters on Symbolic Regression
![Testing Regression](/Testing_GP_Parameters_on_Symbolic_Regression.pdf)
### Classification Analysis
![Breast Cancer Classification](/Classification_of_Breast_Cancer_with_Genetic_Programing_2.pdf)


## Authors

- [@akaJengo](https://github.com/akaJengo)
- [@aidanLarock](https://github.com/aidanLarock)


# go-iforest

[![GoDoc](https://godoc.org/github.com/e-XpertSolutions/go-iforest/iforest?status.png)](http://godoc.org/github.com/e-XpertSolutions/go-iforest/iforest)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-yellow.svg?style=flat)](https://github.com/e-XpertSolutions/go-iforest/blob/master/LICENSE)
[![GoReport](https://goreportcard.com/badge/github.com/e-XpertSolutions/go-iforest)](https://goreportcard.com/report/github.com/e-XpertSolutions/go-iforest)
[![Travis](https://travis-ci.org/e-XpertSolutions/go-iforest.svg?branch=master)](https://travis-ci.org/e-XpertSolutions/go-iforest)
[![cover.run go](https://cover.run/go/github.com/e-XpertSolutions/go-iforest/iforest.svg)](https://cover.run/go/github.com/e-XpertSolutions/go-iforest/iforest)


Go implementation of Isolation Forest algorithm.

Isolation Forest is an unsupervised learning algorithm that is able to detect anomalies (data patterns that differ from normal instances). Detection is performed by recursive data partitioning, which can be represented by a tree structure. At each iteration data is splitted using randomly chosen feature and its value (random number between maximum and minimum value of chosen feature). Due to the fact that anomalies are rare and different from other instances, smaller number of partitions is needed to isolate them. This is equivalent to the path length in created tree. Shorter path means that given instance can be an anomaly. To improve accuracy the ensemble of such trees is created and result is averaged over all trees.

To get more information about algorithm, please refer to this paper: [IFOREST](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).

## Installation

Stable release can be downloaded by issuing the following command:

```
go get github.com/e-XpertSolutions/go-iforest/v2
```

## Usage

This example shows how to use the Isolation Forest. You will need to load the data, initialize iforest with proper parameters and use two functions: Train(), Test() to create the model. First one is used to build the trees, second one to find proper "anomaly threshold" and detect anomalies in given data. After that you can pass new instances to Predict() which result in labeling them as normal "0" or anomaly "1". 
It is possible to use parallel versions of testing and detecting functions - they use multiple go routines to speed up computations.
Created models can be saved and read from the files using Save() and Load() methods.

```go
package main

import(
    "fmt"
    "github.com/e-XpertSolutions/go-iforest/iforest"
)

func main(){

    // input data must be loaded into two dimensional array of the type float64
    // please note: loadData() is some custom function - not included in the
    // library
    var inputData [][]float64
    inputData = loadData("filename")


    // input parameters
    treesNumber := 100
    subsampleSize := 256
    outliersRatio := 0.01
    routinesNumber := 10

    //model initialization
    forest := iforest.NewForest(treesNumber, subsampleSize, outliers)


    //training stage - creating trees
    forest.Train(inputData)

    //testing stage - finding anomalies 
    //Test or TestParaller can be used, concurrent version needs one additional
    // parameter
    forest.Test(inputData)
    forest.TestParallel(inputData, routinesNumber)

    //after testing it is possible to access anomaly scores, anomaly bound 
    // and labels for the input dataset
    threshold := forest.AnomalyBound
    anomalyScores := forest.AnomalyScores
    labelsTest := forest.Labels

    //to get information about new instances pass them to the Predict function
    // to speed up computation use concurrent version of Predict 
    var newData [][]float64
    newData = loadData("someNewInstances")
    labels, scores := forest.Predict(newData)


}
```

## Contributing

Contributions are greatly appreciated. The project follows the typical
[GitHub pull request model](https://help.github.com/articles/using-pull-requests/)
for contribution.


## License

The sources are release under a BSD 3-Clause License. The full terms of that
license can be found in `LICENSE` file of this repository.


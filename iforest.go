package iforest

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
)

// Euler is an Euler's constant as descibed in algorithm specification
const Euler float64 = 0.5772156649

type kv struct {
	Key   int
	Value float64
}

func computeC(n float64) float64 {
	return 2*(math.Log(n-1)+Euler) - ((2 * (n - 1)) / n)
}

// Forest is a base structure for Isolation Forest algorithm. It holds algorithm
// parameters like number of trees, subsampling size, anomalies ratio and
// collection of created trees.
type Forest struct {
	Trees           []Tree
	NbTrees         int
	SubsamplingSize int
	HeightLimit     int
	CSubSampl       float64
	AnomalyScores   map[int]float64
	AnomalyBound    float64
	AnomalyRatio    float64
	Labels          []int
	Trained         bool
	Tested          bool
}

//NewForest initializes Forest structure.
func NewForest(nbTrees, subsamplingSize int, anomalyRatio float64) *Forest {
	f := &Forest{NbTrees: nbTrees, SubsamplingSize: subsamplingSize}
	f.HeightLimit = int(math.Ceil(math.Log2(float64(subsamplingSize))))
	f.Trees = make([]Tree, nbTrees)
	f.AnomalyScores = make(map[int]float64)
	//f.Random = rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	f.AnomalyRatio = anomalyRatio
	n := float64(subsamplingSize)
	f.CSubSampl = computeC(n)
	fmt.Println(f.CSubSampl)
	return f
}

// Train creates the collection of trees in the forest. This is the training
// stage of the algorithm.
func (f *Forest) Train(X [][]float64) {
	mx := make(map[int][]float64, len(X))
	for i := 0; i < len(X); i++ {
		mx[i] = X[i]
	}

	for i := 0; i < f.NbTrees; i++ {
		subsamplesIndicies := f.createSubsamplesWithoutReplacement(mx)
		MaxDepth = f.HeightLimit
		f.Trees[i] = Tree{}
		f.Trees[i].BuildTree(X, subsamplesIndicies)
	}
	f.Trained = true
}

// Test is the algoritm "Evaluating Stage". It computes anomaly scores for the
// dataset (should be used with the same set as in training) and chooses anomaly
// score that will be the bound for detecting anomalies.
func (f *Forest) Test(X [][]float64) error {

	if !f.Trained {
		return errors.New("cannot start testing phase - model has not been trained yet")
	}

	for i := 0; i < len(X); i++ {
		var sumPathLength float64
		sumPathLength = 0.0
		for j := 0; j < len(f.Trees); j++ {
			path := PathLength(X[i], 0, f.Trees[j].Root)
			sumPathLength += path
		}
		averagePath := sumPathLength / float64(len(f.Trees))
		s := 0.5 - math.Pow(2, (-averagePath/f.CSubSampl))
		f.AnomalyScores[i] = s
	}

	sorted := sortMap(f.AnomalyScores)
	anomFloor := int(math.Floor(f.AnomalyRatio * float64(len(X))))
	anomCeil := int(math.Ceil(f.AnomalyRatio * float64(len(X))))
	f.AnomalyBound = (sorted[anomFloor].Value + sorted[anomCeil].Value) / 2

	f.Labels = make([]int, len(X))
	for i := 0; i < len(X); i++ {
		if f.AnomalyScores[i] < f.AnomalyBound {
			f.Labels[i] = 1
		} else {
			f.Labels[i] = 0
		}
	}

	f.Tested = true

	return nil

}

// Predict computes anomaly scores for given dataset and classifies each vector
// as 'normal' or 'anomaly'.
func (f *Forest) Predict(X [][]float64) ([]int, []float64, error) {

	if !f.Trained {
		return nil, nil, errors.New("cannot predict - model has not been trained yet")
	}
	if !f.Tested {
		return nil, nil, errors.New("cannot predict - model has not been tested yet")
	}

	labels := make([]int, len(X))
	scores := make([]float64, len(X))

	for i := 0; i < len(X); i++ {
		var sumPathLength float64
		for j := 0; j < len(f.Trees); j++ {
			path := PathLength(X[i], 0, f.Trees[j].Root)
			sumPathLength += path
		}
		averagePath := sumPathLength / float64(len(f.Trees))
		s := 0.5 - math.Pow(2, (-averagePath/f.CSubSampl))

		scores[i] = s
	}

	for i := 0; i < len(X); i++ {
		if scores[i] < f.AnomalyBound {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	return labels, scores, nil

}

// TestParallel does the same as Test but using multiple go routines
func (f *Forest) TestParallel(X [][]float64, routinesNumber int) error {

	if !f.Trained {
		return errors.New("cannot start testing phase - model has not been trained yet")
	}

	if routinesNumber > len(X) || routinesNumber == 0 {
		return errors.New("number of routines cannot be bigger than nubmer of vectors or equal to 0")
	}
	var CAnomalyScores sync.Map
	var wg sync.WaitGroup
	wg.Add(routinesNumber)
	vectorsPerRoutine := int(len(X) / routinesNumber)

	for j := 0; j < routinesNumber; j++ {
		go f.computeAnomalies(X, j*vectorsPerRoutine, j*vectorsPerRoutine+vectorsPerRoutine, &wg, &CAnomalyScores)
	}
	wg.Wait()

	CAnomalyScores.Range(func(key, value interface{}) bool {
		f.AnomalyScores[key.(int)] = value.(float64)
		return true
	})

	sorted := sortMap(f.AnomalyScores)
	anomFloor := int(math.Floor(f.AnomalyRatio * float64(len(X))))
	anomCeil := int(math.Ceil(f.AnomalyRatio * float64(len(X))))
	f.AnomalyBound = (sorted[anomFloor].Value + sorted[anomCeil].Value) / 2

	f.Labels = make([]int, len(X))
	for i := 0; i < len(X); i++ {
		if f.AnomalyScores[i] < f.AnomalyBound {
			f.Labels[i] = 1
		} else {
			f.Labels[i] = 0
		}
	}
	return nil
}

// PredictParallel computes anomaly scores for given dataset and classifies each vector
// as 'normal' or 'anomaly'. Uses  multiple go routines to make computation faster.
func (f *Forest) PredictParallel(X [][]float64, routinesNumber int) ([]int, []float64, error) {

	if !f.Trained {
		return nil, nil, errors.New("cannot predict - model has not been trained yet")
	}

	if !f.Tested {
		return nil, nil, errors.New("cannot predict - model has not been tested yet")
	}

	if routinesNumber > len(X) || routinesNumber == 0 {
		return nil, nil, errors.New("number of routines cannot be bigger than nubmer of vectors or equal to 0")
	}

	labels := make([]int, len(X))
	scores := make([]float64, len(X))

	var wg sync.WaitGroup
	wg.Add(routinesNumber)
	vectorsPerRoutine := int(len(X) / routinesNumber)

	for j := 0; j < routinesNumber; j++ {
		go func(start, stop int) {
			for i := start; i < stop; i++ {
				var sumPathLength float64
				sumPathLength = 0.0
				for j := 0; j < len(f.Trees); j++ {
					path := PathLength(X[i], 0, f.Trees[j].Root)
					sumPathLength += path

				}
				averagePath := sumPathLength / float64(len(f.Trees))
				s := 0.5 - math.Pow(2, (-averagePath/f.CSubSampl))
				scores[i] = s
			}

			wg.Done()
		}(j*vectorsPerRoutine, j*vectorsPerRoutine+vectorsPerRoutine)
	}
	wg.Wait()

	for i := 0; i < len(X); i++ {
		if scores[i] < f.AnomalyBound {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	return labels, scores, nil

}

// createSubsamples randomly chooses samples from the whole dataset to create
// subset of the size equals to SubsamplingSize. This subset is used in the
// training phase. Using this function one sample may occur several times in
// one subset.
func (f *Forest) createSubsamples(X [][]float64) []int { //use 'without replacement' method (impossible to have two same vectors in one subsample)

	subsamplesIds := make([]int, f.SubsamplingSize)

	for i := 0; i < f.SubsamplingSize; i++ {
		j := rand.Intn(f.SubsamplingSize)
		subsamplesIds[i] = j
	}
	return subsamplesIds
}

func (f *Forest) createSubsamplesWithoutReplacement(X map[int][]float64) []int {

	subsamplesIds := make([]int, f.SubsamplingSize)
	currentSize := 0
	for i := range X {
		subsamplesIds[currentSize] = i
		currentSize++
		if currentSize >= f.SubsamplingSize {
			return subsamplesIds
		}
	}
	return subsamplesIds
}

func (f *Forest) computeAnomalies(X [][]float64, start, stop int, wg *sync.WaitGroup, as *sync.Map) {

	for i := start; i < stop; i++ {
		var sumPathLength float64
		sumPathLength = 0.0
		for j := 0; j < len(f.Trees); j++ {
			path := PathLength(X[i], 0, f.Trees[j].Root)
			sumPathLength += path

		}
		averagePath := sumPathLength / float64(len(f.Trees))
		s := 0.5 - math.Pow(2, (-averagePath/f.CSubSampl))
		as.Store(i, s)
	}

	wg.Done()
}

// sortMap sorts given map in increasing order.
func sortMap(m map[int]float64) []kv {
	var ss []kv
	for k, v := range m {
		ss = append(ss, kv{k, v})
	}

	sort.Slice(ss, func(i, j int) bool {
		return ss[i].Value < ss[j].Value
	})

	return ss
}

// Save saves model in the file
func (f *Forest) Save(path string) error {

	file, err := os.Create(path)
	if err == nil {
		encoder := json.NewEncoder(file)
		err = encoder.Encode(f)
	}
	file.Close()
	return err
}

// Load loads from the file
func (f *Forest) Load(path string) error {
	file, err := os.Open(path)
	if err == nil {
		f.AnomalyScores = make(map[int]float64)
		decoder := json.NewDecoder(file)
		err = decoder.Decode(&f)
	}
	file.Close()

	return err
}

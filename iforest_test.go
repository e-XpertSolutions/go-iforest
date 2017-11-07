package iforest

import (
	"reflect"
	"testing"
)

func Test_sortMap(t *testing.T) {
	tests := []struct {
		m    map[int]float64
		want []kv
	}{
		{m: map[int]float64{0: 1.1, 2: 2, 3: 1.2}, want: []kv{{Key: 0, Value: 1.1}, {Key: 3, Value: 1.2}, {Key: 2, Value: 2}}},
	}
	for _, tt := range tests {

		if got := sortMap(tt.m); !reflect.DeepEqual(got, tt.want) {
			t.Errorf("sortMap() = %v, want %v", got, tt.want)
		}

	}
}

func TestForest_createSubsamplesWithoutReplacement(t *testing.T) {

	tests := []struct {
		f    *Forest
		X    map[int][]float64
		want int
	}{
		{f: NewForest(1, 2, 0.1), X: map[int][]float64{0: {1, 1, 1}, 1: {1, 1, 1}, 2: {1, 1, 1}}, want: 2},
		{f: NewForest(1, 8, 0.1), X: map[int][]float64{0: {1, 1, 1}, 1: {1, 1, 1}, 2: {1, 1, 1}, 3: {1, 2, 3}, 4: {2, 3, 4}, 5: {2, 4, 5}}, want: 8},
	}

	for _, tt := range tests {

		if got := tt.f.createSubsamplesWithoutReplacement(tt.X); len(got) != tt.want {
			t.Errorf("Forest.createSubsamplesWithoutReplacement() = %v, want %v", got, tt.want)
		}

	}
}

func TestForest_createSubsamples(t *testing.T) {

	tests := []struct {
		name string
		f    *Forest
		X    [][]float64
		want int
	}{
		{f: NewForest(1, 2, 0.1), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}, want: 2},
		{f: NewForest(1, 8, 0.1), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 2, 3}, {2, 3, 4}, {2, 4, 5}}, want: 8},
	}
	for _, tt := range tests {

		if got := tt.f.createSubsamples(tt.X); len(got) != tt.want {
			t.Errorf("Forest.createSubsamples() = %v, want %v", got, tt.want)
		}

	}
}

func TestForest_Train(t *testing.T) {
	tests := []struct {
		f *Forest
		X [][]float64
	}{
		{f: NewForest(1, 2, 0.1), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}},
	}
	for _, tt := range tests {

		tt.f.Train(tt.X)

	}
}

func TestForest_Test(t *testing.T) {
	tests := []struct {
		X          [][]float64
		f          *Forest
		trainFirst bool
		wantErr    bool
	}{
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: false, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: true, wantErr: false},
	}
	for _, tt := range tests {

		if tt.trainFirst {
			tt.f.Train(tt.X)
		}

		if err := tt.f.Test(tt.X); (err != nil) != tt.wantErr {
			t.Errorf("Forest.Test() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

func TestForest_Predict(t *testing.T) {

	tests := []struct {
		f          *Forest
		trainFirst bool
		testFirst  bool
		X          [][]float64
		wantErr    bool
	}{
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: false, testFirst: false, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: true, testFirst: false, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, trainFirst: true, testFirst: true, wantErr: false},
	}
	for _, tt := range tests {
		if tt.trainFirst {
			tt.f.Train(tt.X)
		}
		if tt.testFirst {
			tt.f.Test(tt.X)
		}
		_, _, err := tt.f.Predict(tt.X)
		if (err != nil) != tt.wantErr {
			t.Errorf("Forest.Predict() error = %v, wantErr %v", err, tt.wantErr)
			return
		}

	}
}

func TestForest_PredictParallel(t *testing.T) {

	tests := []struct {
		f              *Forest
		X              [][]float64
		routinesNumber int
		trainFirst     bool
		testFirst      bool
		wantErr        bool
	}{
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 2, trainFirst: false, testFirst: false, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 2, trainFirst: true, testFirst: false, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 1, trainFirst: true, testFirst: true, wantErr: false},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 0, trainFirst: true, testFirst: true, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 10, trainFirst: true, testFirst: true, wantErr: true},
	}
	for i, tt := range tests {

		if tt.trainFirst {
			tt.f.Train(tt.X)
		}
		if tt.testFirst {
			tt.f.Test(tt.X)
		}
		_, _, err := tt.f.PredictParallel(tt.X, tt.routinesNumber)
		if (err != nil) != tt.wantErr {
			t.Errorf("%d. Forest.PredictParallel() error = %v, wantErr %v", i, err, tt.wantErr)
			return
		}

	}
}

func TestForest_TestParallel(t *testing.T) {

	tests := []struct {
		f              *Forest
		X              [][]float64
		routinesNumber int
		trainFirst     bool
		wantErr        bool
	}{
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 2, trainFirst: false, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 1, trainFirst: true, wantErr: false},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 0, trainFirst: true, wantErr: true},
		{f: NewForest(1, 4, 0.4), X: [][]float64{{1, 1, 1}, {1, 1, 1}, {10, 10, 10}}, routinesNumber: 10, trainFirst: true, wantErr: true},
	}
	for _, tt := range tests {

		if tt.trainFirst {
			tt.f.Train(tt.X)
		}
		if err := tt.f.TestParallel(tt.X, tt.routinesNumber); (err != nil) != tt.wantErr {
			t.Errorf("Forest.TestParallel() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

func TestForest_Save(t *testing.T) {

	tests := []struct {
		name    string
		f       *Forest
		path    string
		wantErr bool
	}{
		{f: NewForest(1, 2, 0.1), path: "", wantErr: true},
		{f: NewForest(1, 2, 0.1), path: "aaa", wantErr: false},
	}
	for _, tt := range tests {

		if err := tt.f.Save(tt.path); (err != nil) != tt.wantErr {
			t.Errorf("Forest.Save() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

func TestForest_Load(t *testing.T) {

	tests := []struct {
		name    string
		f       *Forest
		path    string
		wantErr bool
	}{
		{f: NewForest(1, 2, 0.1), path: "", wantErr: true},
		{f: NewForest(1, 2, 0.1), path: "xD", wantErr: true},
		{f: NewForest(1, 2, 0.1), path: "aaa", wantErr: false},
	}
	for _, tt := range tests {

		if err := tt.f.Load(tt.path); (err != nil) != tt.wantErr {
			t.Errorf("Forest.Load() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

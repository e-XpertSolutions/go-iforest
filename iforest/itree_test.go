package iforest

import (
	"math"
	"reflect"
	"testing"
)

func Test_splitMatrix(t *testing.T) {
	type args struct {
		X         [][]float64
		split     float64
		attribute int
		ind       []int
	}
	tests := []struct {
		args  args
		want  []int
		want1 []int
	}{
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, split: 2, attribute: 0, ind: []int{0, 1, 2, 3}}, want: []int{0, 1}, want1: []int{2, 3}},
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, split: 3, attribute: 1, ind: []int{0, 1, 2, 3}}, want: []int{0, 1, 2}, want1: []int{3}},
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, split: 6, attribute: 1, ind: []int{0, 1, 2, 3}}, want: []int{0, 1, 2, 3}, want1: []int{}},
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, split: 2, attribute: 0, ind: []int{1, 2}}, want: []int{1}, want1: []int{2}},
	}
	for _, tt := range tests {

		got, got1 := splitMatrix(tt.args.X, tt.args.split, tt.args.attribute, tt.args.ind)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("splitMatrix() got = %v, want %v", got, tt.want)
		}
		if !reflect.DeepEqual(got1, tt.want1) {
			t.Errorf("splitMatrix() got1 = %v, want %v", got1, tt.want1)
		}

	}
}

func Test_findSplit(t *testing.T) {
	type args struct {
		X   [][]float64
		ind []int
		att int
	}
	tests := []struct {
		args  args
		want1 float64
		want2 float64
	}{
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, att: 0, ind: []int{0, 1, 2, 3}}, want1: 1, want2: 3},
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, att: 1, ind: []int{0, 1, 2, 3}}, want1: 2, want2: 5},
	}
	for _, tt := range tests {

		if got := findSplit(tt.args.X, tt.args.ind, tt.args.att); got < tt.want1 || got > tt.want2 {
			t.Errorf("findSplit() = %v, want between %v : %v", got, tt.want1, tt.want2)
		}

	}
}

func Test_findAttribute(t *testing.T) {

	tests := []struct {
		nbAtt int
		want  int
	}{
		{nbAtt: 10, want: 10},
		{nbAtt: 1, want: 1},
		{nbAtt: 5, want: 5},
	}
	for _, tt := range tests {

		if got := findAttribute(tt.nbAtt); got > tt.want {
			t.Errorf("findAttribute() = %v, want %v", got, tt.want)
		}

	}
}

func TestPathLength(t *testing.T) {
	type args struct {
		V          []float64
		pathLength int
	}
	tests := []struct {
		args args
		want float64
	}{
		{args: args{V: []float64{1, 2}, pathLength: 0}, want: 1},
		{args: args{V: []float64{5, 2}, pathLength: 0}, want: 1 + c(5)},
	}
	node := makeSmallTree()
	for _, tt := range tests {
		if got := PathLength(tt.args.V, tt.args.pathLength, node); got != tt.want {
			t.Errorf("PathLength() = %v, want %v", got, tt.want)
		}

	}
}

func makeSmallTree() *Node {
	root := &Node{External: false, Split: 3, Attribute: 0, Size: 6}
	root.Left = &Node{External: true, Size: 1}
	root.Right = &Node{External: true, Size: 5}

	root.C = c(float64(root.Size))
	root.Left.C = c(float64(root.Left.Size))
	root.Right.C = c(float64(root.Right.Size))

	return root
}

func c(n float64) float64 {
	return 2*(math.Log(n-1)+Euler) - ((2 * (n - 1)) / n)
}

func Test_allSame(t *testing.T) {
	type args struct {
		X   [][]float64
		ind []int
	}
	tests := []struct {
		args args
		want bool
	}{
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, ind: []int{0, 1, 2, 3}}, want: false},
		{args: args{X: [][]float64{{1, 2}, {2, 2}, {3, 5}}, ind: []int{0, 1, 2}}, want: false},
		{args: args{X: [][]float64{{1, 2}, {2, 2}, {1, 2}}, ind: []int{0, 2}}, want: true},
		{args: args{X: [][]float64{{2, 2}, {2, 2}, {2, 2}}, ind: []int{0, 1, 2}}, want: true},
	}
	for _, tt := range tests {

		if got := allSame(tt.args.X, tt.args.ind); got != tt.want {
			t.Errorf("allSame() = %v, want %v", got, tt.want)
		}

	}
}

func Test_nextNode(t *testing.T) {
	type args struct {
		X        [][]float64
		indicies []int
		d        int
	}
	tests := []struct {
		args args
		want *Node
	}{
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, indicies: []int{0, 1, 2, 3}, d: 0}, want: &Node{Left: &Node{}, Right: &Node{}, External: false, Size: 4, C: c(float64(4))}},
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, indicies: []int{0, 1}, d: 0}, want: &Node{Left: &Node{}, Right: &Node{}, External: false, Size: 2, C: c(float64(2))}},
	}

	MaxDepth = 2
	for _, tt := range tests {

		if got := nextNode(tt.args.X, tt.args.indicies, tt.args.d); got.C != tt.want.C || got.Left == nil || got.Right == nil || got.External != tt.want.External || got.Size != tt.want.Size {
			t.Errorf("nextNode() = %v, want %v", got, tt.want)
		}

	}
}

func TestTree_BuildTree(t *testing.T) {
	type args struct {
		X   [][]float64
		ids []int
	}
	tests := []struct {
		t       *Tree
		args    args
		wantErr bool
	}{
		{args: args{X: [][]float64{{1, 2}, {1, 2}, {2, 2}, {3, 5}}, ids: []int{0, 1, 2, 3}}, t: &Tree{}, wantErr: false},
	}
	for _, tt := range tests {

		if err := tt.t.BuildTree(tt.args.X, tt.args.ids); (err != nil) != tt.wantErr && tt.t.Root != nil {
			t.Errorf("Tree.BuildTree() error = %v, wantErr %v", err, tt.wantErr)
		}

	}
}

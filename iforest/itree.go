package iforest

import (
	"math"
	"math/rand"
)

// MaxDepth limits the depth of trees, is computed during initialization based
// on the subsampling size
var MaxDepth int

// Tree is base structure for the iTree
type Tree struct {
	Root *Node
}

// Node is a structure for iNode
type Node struct {
	Left      *Node
	Right     *Node
	Split     float64
	Attribute int
	C         float64
	Size      int
	External  bool
}

// BuildTree builds decision tree for given data, attributes and values used for
// splits are choosen randomly
func (t *Tree) BuildTree(X [][]float64, ids []int) error {
	xCols := len(X[0])
	att := findAttribute(xCols)
	split := findSplit(X, ids, att)

	root := &Node{Split: split, Attribute: att, Size: len(ids)}

	indiciesSmaller, indiciesBigger := splitMatrix(X, split, att, ids)

	root.External = false
	root.Left = nextNode(X, indiciesSmaller, 1)
	root.Right = nextNode(X, indiciesBigger, 1)

	t.Root = root

	return nil
}

// nextNode finds the attribute and split value for current node and creates
// next leaves(nodes) of the tree.  It finishes when no more data is available
// or when the tree reaches its maximum depth.
func nextNode(X [][]float64, indicies []int, d int) *Node {
	xCols := len(X[0])

	var c float64
	if len(indicies) > 1 {
		c = computeC(float64(len(indicies)))
	}

	if len(indicies) <= 1 || d >= MaxDepth {
		return &Node{External: true, Size: len(indicies), C: c}
	}

	att := findAttribute(xCols)
	split := findSplit(X, indicies, att)

	newNode := &Node{Attribute: att, Split: split, External: false, Size: len(indicies), C: c}

	indiciesSmaller, indiciesBigger := splitMatrix(X, split, att, indicies)
	newNode.Left = nextNode(X, indiciesSmaller, d+1)

	newNode.Right = nextNode(X, indiciesBigger, d+1)

	return newNode

}

//recursive version of PathLength function
/* func PathLength(V []float64, pathLength int, node *Node) float64 {
	//if node.External {
	if node.Size <= 1 {
		return float64(pathLength)
	}
	if node.External {
		return float64(pathLength) + node.C //computeC(float64(node.Size))
	}
	//}
	if V[node.Attribute] < node.Split {
		return PathLength(V, pathLength+1, node.Left)
	}
	return PathLength(V, pathLength+1, node.Right)

} */

// PathLength computes the length of the path for given data vector.
// The result is number of edges from the root to terminating node
// plus adjustment value - c(Size) as described in algorithm specification.
func PathLength(V []float64, pathLength int, node *Node) float64 {

	var currentNode, nextNode *Node
	currentNode = node

	for {
		pathLength++
		if V[currentNode.Attribute] < currentNode.Split {
			nextNode = currentNode.Left
		} else {
			nextNode = currentNode.Right
		}
		currentNode = nextNode

		if currentNode.Size <= 1 {

			return float64(pathLength)
		}
		if currentNode.External {

			return float64(pathLength) + currentNode.C
		}

	}

}

// splitMatrix divides matrix on two parts based on choosen attribute and split
// value
func splitMatrix(X [][]float64, split float64, attribute int, ind []int) ([]int, []int) {

	smaller := make([]int, 0)
	bigger := make([]int, 0)

	for _, val := range ind {
		if X[val][attribute] < split {
			smaller = append(smaller, val)
		} else {
			bigger = append(bigger, val)
		}
	}

	return smaller, bigger
}

// findAttribute randomly choose the attribute for the split
func findAttribute(nbAtt int) int {
	return rand.Intn(nbAtt)

}

// findSplit randomly choose value of the split. This value is always between
// lowest and highest value among the attribute values.
func findSplit(X [][]float64, ind []int, att int) float64 {
	max := -math.MaxFloat64
	min := math.MaxFloat64

	for _, val := range ind {
		if X[val][att] <= min {
			min = X[val][att]
		}
		if X[val][att] >= max {
			max = X[val][att]
		}

	}
	return min + (max-min)*rand.Float64()

}

// allSame checks if all vectors in given matrix are the same
func allSame(X [][]float64, ind []int) bool {
	for _, el := range ind[1:] {
		var different bool
		for i := 0; i < len(X[el]); i++ {

			if X[el][i] != X[ind[0]][i] {
				different = true
			}
		}
		if different {
			return false
		}
	}
	return true
}

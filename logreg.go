package main

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/hdf5"
	"fmt"
)

/**
def initialize_w_zeroes(dim): #for most networks you would init with random values
    zeroes = np.zeros((dim, 1))
    b = 0
    assert(zeroes.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int)) #i don't like python.
    return zeroes, b

**/

func initWZeros(dim int) *mat.VecDense {
	zeroSlice := make([]float64, dim) //initialize a slice filled with zeros
	initWeights := mat.NewVecDense(dim, zeroSlice)
	return initWeights
}

func main() {
	fmt.Println(initWZeros(6))
}

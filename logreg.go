package main
//references https://mark.douthwaite.io/a-brief-intro-tohdf5/

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

func loadImages(test_file string, training_file string, image []float64, height int, width int) {
	test_data, err := hdf5.CreateFile(test_file, hdf5.F_ACC_TRUNC)
	if err != nil {
		panic(err)
	}
	defer test_data.Close()

	dims := []uint{uint(len(image))}
	space, err := hdf5.CreateSimpleDataspace(dims, nil) //make space for your data
	if err != nil {
		panic("failed to create simple dataspace")
	}
	dtype, err := hdf5.NewDatatypeFromValue(image[0]) //get the type of data in this array.
	if err != nil {
		panic("failed to create dtype")
	}

	dataset, err := test_data.CreateDataset("data", dtype, space) //use everything we made to create a data set
	err = dataset.Write(&image)
	if err != nil {
		panic("failed to write image data into the dataset")
	}''


}


func initWZeros(dim int) *mat.VecDense {
	zeroSlice := make([]float64, dim) //initialize a slice filled with zeros
	initWeights := mat.NewVecDense(dim, zeroSlice)
	return initWeights
}

func main() {
	fmt.Println(initWZeros(6))
}

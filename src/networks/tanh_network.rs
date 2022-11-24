use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::networks::neural_network::*;

/**Defining the Neural Network*/
pub struct TanhNetwork {
    //The matrices in the Network
    //weights
    hidden_weights: Array2<f64>,
    output_weights: Array2<f64>,
    //bias
    hidden_bias: Array2<f64>,
    output_bias: Array2<f64>,
    //layers
    hidden_layer: Array2<f64>,
    output_layer: Array2<f64>,
}

/**Implementing the unique activation functions for this implementation of the network */
impl TanhNetwork {
    /**Constructor-like function that initializes and returns a network */
    pub fn new() -> Self {
        //initializes each layer and its corresponding weights and biases
        //the hidden and output layers themselves begin empty
        Self {
            //initial values of weights are randomized
            //hidden
            hidden_layer: Array::default((0, 0)),
            hidden_weights: Array::random((20, 784), Uniform::new(-0.5, 0.5)),
            hidden_bias: Array::zeros((20, 1)),
            //output
            output_layer: Array::default((0, 0)),
            output_weights: Array::random((10, 20), Uniform::new(-0.5, 0.5)),
            output_bias: Array::zeros((10, 1)),
        }
    }
}

/**Implementing the neural net trait for the network using Tanh for the activation function */
impl NeuralNetwork for TanhNetwork {
    /**Implementing the activation function as the tanh function of x*/
    fn activation(x: Array2<f64>) -> Array2<f64> {
        //calling tanh(x) on each element in mapv
        x.mapv(|v| v.tanh())
    }

    /**Implementing the derivative of tanh */
    fn activ_deriv(x: &Array2<f64>) -> Array2<f64> {
        //d/dx tanh(x) = 1 - tanh^2(x)
        x.mapv(|v| 1.0 - (v.tanh()).powi(2))
    }

    /**Setter and getter functions for each array */
    //Getters
    fn get_hid_layer(&self) -> Array2<f64> {
        self.hidden_layer.to_owned()
    }
    fn get_hid_weights(&self) -> Array2<f64> {
        self.hidden_weights.to_owned()
    }
    fn get_hid_bias(&self) -> Array2<f64> {
        self.hidden_bias.to_owned()
    }
    fn get_out_layer(&self) -> Array2<f64> {
        self.output_layer.to_owned()
    }
    fn get_out_weights(&self) -> Array2<f64> {
        self.output_weights.to_owned()
    }
    fn get_out_bias(&self) -> Array2<f64> {
        self.output_bias.to_owned()
    }
    //Setters
    fn set_hid_layer(&mut self, x: Array2<f64>) {
        self.hidden_layer = x;
    }
    fn set_hid_weights(&mut self, x: Array2<f64>) {
        self.hidden_weights = x;
    }
    fn set_hid_bias(&mut self, x: Array2<f64>) {
        self.hidden_bias = x;
    }
    fn set_out_layer(&mut self, x: Array2<f64>) {
        self.output_layer = x;
    }
    fn set_out_weights(&mut self, x: Array2<f64>) {
        self.output_weights = x;
    }
    fn set_out_bias(&mut self, x: Array2<f64>) {
        self.output_bias = x;
    }
}

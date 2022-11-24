use ndarray::{Array, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::networks::neural_network::*;

/**Defining the Neural Network*/
pub struct ReluNetwork {
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
impl ReluNetwork {
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

    /**The softmax activation function to be used on the output layer */
    fn softmax(x: Array2<f64>) -> Array2<f64> {
        //subtracting by max value to prevent overflow (NaN values)
        let max = x.column(0).into_iter().fold(f64::MIN, |a, &b| a.max(b));
        let exp = x.mapv(|v| (v - max).exp());
        //println!("{}", exp);
        &exp / exp.sum_axis(Axis(0))
    }
}

/**Implementing the neural net trait for the network using Leaky ReLU and Softmax for the activation functions */
impl NeuralNetwork for ReluNetwork {
    /**Implementing the activation function as ReLU(x), to be used on the hidden layer */
    fn activation(x: Array2<f64>) -> Array2<f64> {
        //Using a variant of ReLU called leaky ReLU,
        //which returns x or 0.01x
        x.mapv(|v| if v > 0.0 { v } else { v * 0.01 })
    }

    /**Implementing as the derivative of ReLU(x) */
    fn activ_deriv(x: &Array2<f64>) -> Array2<f64> {
        //Leaky ReLU's derivative is 1 if x is positive, 0.01 otherwise
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.01 })
    }

    /**Overriding the forward propogation function for the use of ReLU and Softmax */
    fn forward_propagation(&mut self, img: &Array2<f64>) {
        //calculating the hidden layer matrix in h_calc
        let hid_calc = Self::get_hid_weights(&self).dot(img) + &Self::get_hid_bias(&self);
        //applying the activation function sigmoid
        Self::set_hid_layer(self, Self::activation(hid_calc));

        //calculating the output layer matrix
        let out_calc = Self::get_out_weights(&self).dot(&Self::get_hid_layer(&self))
            + &Self::get_out_bias(&self);
        //applying activation function
        Self::set_out_layer(self, Self::softmax(out_calc));
    }

    /**Implements backwards propagation using the input matrices*/
    fn back_propagation(&mut self, img: &Array2<f64>, lab: &Array2<f64>, lr: f64) {
        //Working backwards from the output to the hidden layer
        //delta_out calculates the output error using the derivative of the cost function
        let delta_out = &Self::get_out_layer(&self) - lab;
        Self::set_out_weights(
            self,
            &Self::get_out_weights(&self) + (-lr * &delta_out.dot(&Self::get_hid_layer(&self).t())),
        );
        Self::set_out_bias(self, &Self::get_out_bias(&self) + (-lr * &delta_out));

        //delta_hid is calculated using the derivative of the activation function.
        let delta_hid = Self::get_out_weights(&self).t().dot(&delta_out)
            * Self::activ_deriv(&Self::get_hid_layer(&self));
        Self::set_hid_weights(
            self,
            &Self::get_hid_weights(&self) + (-lr * delta_hid.dot(&img.t())),
        );
        Self::set_hid_bias(self, Self::get_hid_bias(&self) + (-lr * delta_hid));
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

use ndarray::{Array1, Array2, Zip};

/**A trait in rust is similar to interfaces in java.
 * This trait describes a base for the neural network struct */
pub trait NeuralNetwork {
    /**Getter functions */
    //hidden & output layers
    fn get_hid_layer(&self) -> Array2<f64>;
    fn get_hid_weights(&self) -> Array2<f64>;
    fn get_hid_bias(&self) -> Array2<f64>;
    fn get_out_layer(&self) -> Array2<f64>;
    fn get_out_weights(&self) -> Array2<f64>;
    fn get_out_bias(&self) -> Array2<f64>;

    /**Setter functions */
    fn set_hid_layer(&mut self, x: Array2<f64>);
    fn set_hid_weights(&mut self, x: Array2<f64>);
    fn set_hid_bias(&mut self, x: Array2<f64>);
    fn set_out_layer(&mut self, x: Array2<f64>);
    fn set_out_weights(&mut self, x: Array2<f64>);
    fn set_out_bias(&mut self, x: Array2<f64>);

    /**Activation function and its derivative */
    fn activation(x: Array2<f64>) -> Array2<f64>;
    fn activ_deriv(x: &Array2<f64>) -> Array2<f64>;

    /**Optimizes the network, using input matrices x and y, the number of iterations,
     * and the learning rate. */
    fn gradient_descent(&mut self, x: &Array2<f64>, y: &Array2<f64>, epochs: i32, learn_rate: f64) {
        //counter var for number of correct outputs
        let mut correct = 0;
        //for each epoch
        for i in 0..epochs {
            Zip::from(x.rows()).and(y.rows()).for_each(|image, label| {
                //acquiring each image & label from x & y,
                //then changing their shape for later calculations
                let img = image.into_shape((image.len(), 1)).unwrap().to_owned();
                let lab = label.into_shape((label.len(), 1)).unwrap().to_owned();
                //forward propogation assigns values to each matrix
                self.forward_propagation(&img);

                //error calculation, if the current output is correct
                correct += if_correct(&Self::get_out_layer(&self), &lab) as u32;

                //back propagating, modifies each matrix
                self.back_propagation(&img, &lab, learn_rate);
            });
            println!("After Epoch {}:", i + 1);
            //print accuracy
            let acc = get_percentage(correct, x.dim().0);
            println!("Accuracy: {:.3}%", acc);
            //resetting correct counter
            correct = 0;
        }
    }

    /**Implements forward propagation using an input matrix */
    fn forward_propagation(&mut self, img: &Array2<f64>) {
        //calculating the hidden layer matrix in h_calc
        let hid_calc = Self::get_hid_weights(&self).dot(img) + &Self::get_hid_bias(&self);
        //applying the activation function sigmoid
        Self::set_hid_layer(self, Self::activation(hid_calc));

        //calculating the output layer matrix
        let out_calc = Self::get_out_weights(&self).dot(&Self::get_hid_layer(&self))
            + &Self::get_out_bias(&self);
        //applying activation function
        Self::set_out_layer(self, Self::activation(out_calc));
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

    /**The network will guess what digit the image is */
    fn make_guess(&mut self, image: &Array1<f64>) -> usize {
        //to make a guess, the network must forward propogate once with the img
        let img = image.to_shape((image.len(), 1)).unwrap().to_owned();
        self.forward_propagation(&img);
        argmax(&Self::get_out_layer(&self).column(0).to_owned())
    }
}

/**Used to determine if the output array's most likely predicition is correct*/
pub fn if_correct(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    argmax(&a.column(0).to_owned()) == argmax(&b.column(0).to_owned())
}

/**returns the percentage of correct guesses during training */
pub fn get_percentage(correct: u32, total: usize) -> f64 {
    correct as f64 / total as f64 * 100.0
}

/**custom implementation of numpy's argmax function, which takes an array and
 * outputs the index where the maximum values occurs.
*/
pub fn argmax(m: &Array1<f64>) -> usize {
    let max = m
        .into_iter()
        .enumerate()
        //for each index-element tuple a & b
        .fold((0, f64::MIN), |a, b| {
            //if a's val > b's val, set a as new max
            if a.1 > *b.1 {
                (a.0, a.1)
            //otherwise b is new max
            } else {
                (b.0, *b.1)
            }
        });
    //returning the index portion of the tuple
    max.0
}

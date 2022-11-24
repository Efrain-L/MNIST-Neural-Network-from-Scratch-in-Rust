mod read_data;
use read_data::*;

mod networks;
use crate::networks::{
    neural_network::NeuralNetwork, relu_network::ReluNetwork, sigmoid_network::SigmoidNetwork,
    tanh_network::TanhNetwork,
};

fn main() {
    println!("This program will train Neural Networks on the MNIST Data Set using different activation functions with the same parameters for comparison.");
    let quantity = 10000;
    println!("Loading in training data from file ({} images).", quantity);
    let (x_train, y_train) = get_training_data(quantity);
    //parameters
    let epochs = 3;
    let learn_rate = 0.01;
    println!(
        "Training data and parameters set. ({} epochs, {} learn rate).\n",
        epochs, learn_rate
    );

    //Network using Sigmoid for its activation function
    println!("Training Neural Network 1 using Sigmoid...");
    //creating
    let mut sig_net = SigmoidNetwork::new();
    //training
    sig_net.gradient_descent(&x_train, &y_train, epochs, learn_rate);
    println!("Sigmoid Network training complete.\n");

    //Network using Tanh for its activation function
    println!("Training Neural Network 2 using Tanh...");
    //creating
    let mut tanh_net = TanhNetwork::new();
    //training
    tanh_net.gradient_descent(&x_train, &y_train, epochs, learn_rate);
    println!("Tanh Network training complete.\n");

    //Network using ReLU for its activation function
    println!("Training Neural Network 3 using ReLU...");
    //creating
    let mut relu_net = ReluNetwork::new();
    //training
    relu_net.gradient_descent(&x_train, &y_train, epochs, learn_rate);
    println!("ReLU Network training complete.\n");

    //Test network by making guesses on test set which it has not been trained on
    println!("Acquiring testing data from file.");
    let (x_test, y_test) = get_testing_data();
    loop {
        //Select an image from set
        println!("\nEnter 0 to 9999 to test an image from test set (or -1 to exit):");
        let mut input = String::new();
        let _read = std::io::stdin()
            .read_line(&mut input)
            .expect("failed to read input");
        let idx: i64 = input.trim().parse().unwrap();

        //exit loop condition
        if idx == -1 {
            println!("Exiting...");
            break;
        }

        //invalid input check
        if idx < 0 || idx > 9999 {
            println!("Invalid input try again.");
            continue;
        }

        //Network will guess the image
        println!("Testing image #{} in test set:", idx);
        let image = x_test.row(idx as usize).to_owned();
        let label = y_test.get(idx as usize).unwrap();

        //showing the image chosen
        show_image(&image);

        //Making the guess
        let sig_guess = sig_net.make_guess(&image);
        let tanh_guess = tanh_net.make_guess(&image);
        let relu_guess = relu_net.make_guess(&image);
        println!("The Sigmoid Network guessed that this is a: {}", sig_guess);
        println!("The Tanh Network guessed that this is a: {}", tanh_guess);
        println!("The ReLU Network guessed that this is a: {}", relu_guess);
        println!("The digit is actually a: {}", label);
    }
}

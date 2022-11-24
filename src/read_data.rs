use csv::ReaderBuilder;
use ndarray::{s, Array, Array1, Array2};
use ndarray_csv::Array2Reader;
use raster::{self, Color};
use std::error::Error;
use std::fs::File;

/*Reads the data from a csv file into a 2-D Array, may return error*/
pub fn read_csv(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file_path = File::open(path)?;
    let mut reader = ReaderBuilder::new().from_reader(file_path);
    //reading file into 2-D array
    let data: Array2<f64> = reader.deserialize_array2_dynamic()?;
    //return data if there is no error
    Ok(data)
}

/**One hot encoding involves taking an array with categorical data, and
 * encoding that categorical portion into 1's and 0's */
pub fn one_hot_encode(y: &Array1<f64>) -> Array2<f64> {
    //the maximum value in this case will be the number of categories
    let max = y.fold(f64::MIN, |a, &b| a.max(b)) as usize;
    let mut one_hot_y = Array::zeros((y.len(), max + 1));
    //starting with 0's, place 1's in corresponding locations
    for i in 0..y.len() {
        let y_val = y[i] as usize;
        one_hot_y[[i, y_val]] = 1.0;
    }
    one_hot_y
}

/**Returns the training data in two arrays, x containing the pixel values, and y the label
 * size - How many images to retrieve from the data set
*/
pub fn get_training_data(size: usize) -> (Array2<f64>, Array2<f64>) {
    //acquiring train data
    let train_file = r".\mnist_data\mnist_train.csv";
    let train_data = read_csv(train_file)
        .expect("Could not read csv.")
        .t()
        .to_owned();
    let y: Array1<f64> = train_data.slice(s![0, ..size]).to_owned();
    //encoding training data by labels (eg. 0-9 for each digit)
    let o_h_y = one_hot_encode(&y);
    let mut x: Array2<f64> = train_data
        .slice(s![1..train_data.dim().0, ..size])
        .t()
        .to_owned();
    //dividing by max pixel value to convert values to be between 0 and 1
    x /= 255.0;
    (x, o_h_y)
}

/**Returns testing data in two arrays, x containing the pixel values, and y the label */
pub fn get_testing_data() -> (Array2<f64>, Array1<f64>) {
    //acquiring test data
    let test_file = r".\mnist_data\mnist_test.csv";
    let test_data = read_csv(test_file)
        .expect("Could not read csv.")
        .t()
        .to_owned();
    //seperating data into x and y axes
    let y: Array1<f64> = test_data.slice(s![0, ..]).to_owned();
    let mut x: Array2<f64> = test_data.slice(s![1..test_data.dim().0, ..]).t().to_owned();
    //converting to 0 to 1 interval
    x /= 255.0;
    (x, y)
}

/**Generates an image of the selected digit using the pixel data from its array */
pub fn show_image(img: &Array1<f64>) {
    //starting with a blank image, all images in the MNIST set are 28x28.
    let mut result = raster::Image::blank(28, 28);
    let mut p = 0;
    for i in 0..result.height {
        for j in 0..result.width {
            //retrieving the original rgb value by multiplying by 255.
            let rgb_val = (img.get(p).unwrap() * 255.0) as u8;
            result
                .set_pixel(j, i, Color::rgba(rgb_val, rgb_val, rgb_val, 255))
                .unwrap();
            p += 1;
        }
    }
    raster::save(&result, "selected_img.png").unwrap();
}

# Model 1 Description
This neural network is composed of four convolution layers, one flattening layer, and two densely connected layers of neurons. On our test dataset, it has a total accuracy of 97.96%, with the misclassifications being stored in the [misfire directory](misfires).

## Layer Description
| Layer Type | Description |
|:----------:|:-----------:|
| Input | This layer represents the input image. This picture is a 28x28 array of values between 0 and 255. |
| Convolution | This layer applies 12 filters to the original image |
| Convolution | This layer applies 24 filters to the output of the previous layer |
| Convolution | This layer applies 48 filters to the output of the previous layer |
| Convolution | This layer applies 96 filters to the output of the previous layer |
| Flatten | This layer flattens all of the outputs of the layers into a single vector that can be evaluated by our layer of densely connected layers |
| Dense | This layer uses densely-connected neurons to make final classifications on our image |
| Output | This layer uses softmax activation to calculate the probabilities of each class for this image |

## Evaluation
When we test this model on our test dataset, we find that it has an accuracy of 97.96%! There are 204 misclassifications, and those are logged in the misfire directory along with the probabilities.

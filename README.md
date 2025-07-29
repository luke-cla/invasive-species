# Invasive Plant Species Classifier

This project uses machine learning to identify invasive plant species from photographs. It was developed under the mentorship of Dr. Nathan Green at Marymount University, as part of the NSF S-STEM Scholars program (Grant #2221147). The goal is to support local conservation by building an image classification system that non-experts can use in the field.

The long-term vision is a mobile app that enables real-time species recognition through smartphone photos. This would help communities, students, and volunteers detect invasive plants early and contribute to ecological monitoring efforts.

## Species Covered
The model classifies four invasive species prevalent in Northern Virginia:

- **Bush Honeysuckle** (*Lonicera spp.*)
- **English Ivy** (*Hedera helix*)
- **Mile-a-Minute Vine** (*Persicaria perfoliata*)
- **Porcelain-Berry** (*Ampelopsis brevipedunculata*)

Species were chosen based on environmental reports from the Reston National Study Group.

## Dataset
- Over 1,000 images collected from sources
  - [iNaturalist.org](https://www.inaturalist.org/)
- Images were labeled and sorted by species

Due to size, the dataset is not included in this repo. You can recreate it using this folder structure:

```
data/
  honeysuckle/
  english_ivy/
  mile_a_minute/
  porcelain_berry/
```

## Workflow Summary
1. Images are preprocessed using OpenCV - resized, equalized, and normalized
2. Data is split into training and test sets
3. A CNN is trained using TensorFlow and Keras
4. The model outputs accuracy and a confusion matrix
5. Once trained, the model can be used to classify new plant images using 'predict.py'

Model structure and comments are included in `main.py`.

## Results
- **Test Accuracy:** ~80%
- **Best Performance:** Mile-a-Minute and Porcelain-Berry
- **Challenges:** Honeysuckle had lower recall due to limited data

## Next Steps
To improve the model's performance and prepare it for real-world deployment, the following steps are planned:
1. Expand the dataset to include more images and increase class balance
2. Explore data augmentation to solve lack of images available
3. Integrate a native plant dataset (already assembled) to allow for invasive vs non-invasive classification
4. Separate Honeysuckle variants into their own distinct classes, since the current dataset combines multiple species into one, which may be confusing the model
5. Add additional invasive species to improve geographical coverage
6. Evaluate the model on real-world photos to test performance in the wild
7. Develop a mobile app prototype for public use

## How to Run Using the Model that is Already Trained
If you want to test the project without training anything:

1. Clone the repo and open the folder:
   `git clone https://github.com/luke-cla/invasive-species.git && cd invasive-species`
2. Install the required packages: `pip install -r requirements.txt`
3. Open `predict.py` and find the line that sets the image path: `image_path = r"sampleimagepathhere.jpg"`
   Change this line if you want to test a different image (just make sure the image is in the `samples/` folder).
4. Run the prediction: `python predict.py`

This will:
- Load the image
- Use the trained model (`invasive_species_classifier.h5`) to predict the species
- Print the predicted class and confidence level

## How to Run Using Your Own Dataset
1. Make sure you're using **Python 3.7 to 3.10**. TensorFlow does not support newer versions on Windows.
1. Clone the repo and open the folder:
   `git clone https://github.com/luke-cla/invasive-species.git && cd invasive-species`
2. Install required packages: `pip install -r requirements.txt`
3. Add your dataset using the folder structure shown above under "Dataset"
4. Run the script to train the model: `python main.py`

This will train a CNN on your dataset and save the model as `invasive_species_classifier.h5`.
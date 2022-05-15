import splitfolders

input_folder = "Flowers_dataset"
output = "processed_data"
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.6, .2, .2))

## Get Data
The get all the dataset used in this research, please run the following commands.
```
kaggle datasets download mateuszbuda/lgg-mri-segmentation
kaggle datasets download kmader/nih-deeplesion-subset
kaggle datasets download nikhilpandey360/chest-xray-masks-and-labels

unzip lgg-mri-segmentation.zip -d ./data/brain-lesion-seg
rm chest-xray-masks-and-labels.zip

unzip chest-xray-masks-and-labels.zip -d ./data/chest-xray-seg
rm lgg-mri-segmentation.zip

unzip nih-deeplesion-subset.zip -d ./data/brain-lesion-seg
rm nih-deeplesion-subset.zip
```
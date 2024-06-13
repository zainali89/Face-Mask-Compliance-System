# Face-Mask-Compliance-System

Our AI-driven Face Mask Detection aim is to leverage the YOLO (You Only Look Once) algorithm to enhance safety compliance in key public areas. This innovative system uses real-time video analysis to accurately detect and verify face mask usage in environments such as hospitals and schools. By utilizing advanced object detection and deep learning, the solution efficiently processes high-traffic video feeds, ensuring rapid and reliable mask compliance checks. This automation significantly reduces the need for manual monitoring, supporting stricter adherence to health and safety protocols.

### Data Augmentation Technique

The image shows different augmented versions of a person wearing a mask. These augmentations include transformations like rotation, zoom, and horizontal flip, which are used to increase the diversity of the training data. Data augmentation helps improve the model's robustness and ability to generalize by providing it with varied representations of the same image during training.
![data_aug_mask](https://github.com/zainali89/Face-Mask-Compliance-System/assets/75775907/691941be-c002-4bf3-a4c4-dfc4b92def53)


### Confusion Matrix

The confusion matrix visualizes the performance of the face mask detection model on test data, showing true positives, false negatives, false positives, and true negatives. It indicates that the model correctly identified 49 out of 50 instances of people without masks and 48 out of 50 instances of people with masks. The classification report below the matrix details the precision, recall, and F1-score for each class (0: No Mask, 1: Mask), showing high accuracy and balanced performance across both classes.

![cm_mask](https://github.com/zainali89/Face-Mask-Compliance-System/assets/75775907/547cdadd-c4e7-4454-ac98-8544c376f872)

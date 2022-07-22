# Gesture-Controlled Game
Using machine learning to create a game controlled by hand gestures!


Currently, I am using a publicly available, pre-trained model called MediaPipe Hands as the image encoding backbone
(extracting landmarks from hands) and training a support vector classifier using the extracted landmarks. This setup
is not only accurate but also quite efficient (since MediaPipe does a good job optimizing their model!) The current
"game" is barely a game and only used for testing purposes.

## Current to-do list:
- [ ] Provide easy way to download dataset
- [X] Revamp dataset collection script
- [X] Collect new dataset with three actions
- [ ] Improve game graphics
- [ ] Frame-independent movement
- [ ] Revamp movement (simpler, smoother = better)
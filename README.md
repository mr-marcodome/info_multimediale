# Re-Identification

This is a repository for the IM students. Here, students will insert some scripts for Re-Identification.

```
Run the follow python script for people tracking. It creates csv files with all information about the oni file.
```sh
$ python2 src/tracking/features.py --v video.oni
```
Launch features for generate a single csv file wich contains the feature information about a single subject.
To perform feature tracking to many subject video, edit and run training.sh to generate many csv files.

For reidentification of people: 
1- edit training.sh with the single person video filenames, and run it.
2- launch feature.py on the collective test video to generate the feature vector of all people in collective situation.
3- edit and launch svm.py to perform reidentification of subject starting from information collected.

Good test! Have Fun!

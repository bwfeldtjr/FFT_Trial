# FFT_Trial
3-26-19: Got the one plot in KM_Class to work. I started working on plotting that density plot you showed me but it is proving to be much more difficult than plotting lines. I also found some more lines within loops that didn't need to be there, so on my computer it can now run both PSD_Class and KM_Class in under 10 seconds, which is a vast improvement.

3-25-19: KM Class now has all but one plot using ggplot (the linear regression one is giving me a hard time). I've also just removed or improved some redundancies or unecessary lines throughout the KM Class and PSD Class

3-22-19: All of the plots on the PSD Class plot on ggplot and pyplot just so we can see both. I still need to add legends (surprisingly difficult). I'm going to keep working on it after the K-State game is over to try to get the KM Class to be all on ggplot by the end of today.

3-7-19: Currently I am just cleaning up some of the code in the KM class: removing redundancies, making certain things their own functions, deleting variables that were defined as empty lists then never used. I'm also starting to practice with ggplot. I haven't tried to implement it in any of these yet, but I'm confident the transistion from pyplot will not be too difficult. 

Here's where i'll be working on the class

So far the Corr, PSD, and Taylor functions are all up and running fine. They use random trajectories from the csv file to determine the functions they are plotting

The KM Class is built. I still need to go through and clean it up a little bit more. I still have not got to implementing ggplot, and the class took most of my time. 

The new file, TestClass, runs the other two classes, KM and PSD, using the input (currently hard coded) from the user. Creating custom plotter is the next step.

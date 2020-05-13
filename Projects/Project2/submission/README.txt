StanProject.ipynb is the Jupyter Notebook that I primarily used to generate all plots and results you see in the report

StanClasses.py contains my alternative API to PyStan. I use the classes in StanClasses to encapsulate each model types. 
The API resembles Sklearn, so think of them as Sklearn models. The classes reads Stan code from external files:
    - sm_ard.stan (ARD model)
    - sm_nmf.stan (Non-negative model)
    - sm_normal.stan (Normal model)
    - sm_simple.stan (this one is not really used)

utils.py contains various utility functions used throughout the project. StanClasses uses it as well. 

Have a nice day

- Naphat
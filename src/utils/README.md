# `utils` 

The utils module is a catch all for various functions and classes that assist with things like debugging or other non-categorical functions. One important class in this module is the timer struct which can be used to time specific functions in the solver. A user can use it as follows:

1. Include `utils.h` in the file that you wish to add the timer to.
2. Declare the timer `Utils::Timer <timer_name>("<printed statemen for timer>")` just before the function you wish to time.
3. Call `<time_name>.end_timer()` after the function you wish to time. 
4. The amount of time it took for the function to run will be displayed on either the command line or the job's output file.

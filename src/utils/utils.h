#ifndef DG_UTILS_H
#define DG_UTILS_H

#include <string>
#include <iostream>
#include <chrono>
#include "toml11/toml.hpp"

namespace Utils {

/*! \brief Return the content of the file in the form of a string
 *
 * @param fname the file name
 * @return The content of the file as a string
 */
std::string get_file_content(const char *fname);

/*! \brief Performs a basic validity check of the input file
 *
 * This function does a preliminary basic check of the validity of the input file. Anything that
 * can be check prior to instantiating other objects (e.g. equation objects) should be checked
 * here. If something is not valid, an InputException is thrown and the code simply exits.
 *
 * @param input
 */
void check_input_file(const toml::value &input);

/*! \brief Create directories required by the run as specified in the input file
 *
 * This function creates the required directories namely for the different outputs of the code
 * (text outputs, solution files, restart files, etc.).
 * It will only create what is necessary.
 *
 * @param input parsed input information
 */
void create_required_directories(const toml::value &input);

/*! \brief Construct a file name with the iteration number in it
 *
 * @param prefix
 * @param iter
 * @return
 */
std::string get_fname_with_iter(const char *prefix, const char *suffix, const int iter);

/*! \brief Construct a file name with the iteration number and sub-region ID in it
 *
 * @param prefix
 * @param suffix
 * @param iter
 * @param srID
 * @return
 */
std::string get_fname_with_iter_srID(
    const char *prefix,
    const char *suffix,
    const unsigned iter,
    const unsigned srID);

/*! \brief Helper function to get an option from CL with a flag
 *
 * Example of usage:
 * char *option = get_option(argv, argv+argc, "-o");
 *
 * @param begin
 * @param end
 * @param option
 * @return
 */
char * get_option(char ** begin, char ** end, const std::string & option);

/*! \brief Helper function to check if an option in the CL exists
 *
 * @param begin
 * @param end
 * @param option
 * @return
 */
bool exist_option(char** begin, char** end, const std::string& option);

/*! \brief Check intersection between 2 sorted ranges
 *
 * Taken from https://stackoverflow.com/questions/46770028/is-it-possible-to-use-stdset-intersection-to-check-if-two-sets-have-any-elem
 *
 * @tparam I1
 * @tparam I2
 * @param first1
 * @param last1
 * @param first2
 * @param last2
 * @return
 */
template <class I1, class I2>
bool have_common_element(I1 first1, I1 last1, I2 first2, I2 last2) {
    while (first1 != last1 && first2 != last2) {
        if (*first1 < *first2)
            ++first1;
        else if (*first2 < *first1)
            ++first2;
        else
            return true;
    }
    return false;
}

/*! \brief Print the code's header text
 */
void print_header();

struct Timer {
    
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<double> duration;

    const std::string timer_id;

    Timer(const std::string& timer_id) : timer_id{timer_id}{
        start = std::chrono::steady_clock::now();
    }

    void end_timer() {
        end = std::chrono::steady_clock::now();
        duration = end - start;

        double sec = duration.count();
        std::cout << timer_id << " took " << sec << "s " << std::endl; 
    }
};

} // end namespace Utils

#endif //DG_UTILS_H

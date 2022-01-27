//
// Created by kihiro on 2/19/20.
//

#ifndef DG_MY_EXCEPTIONS_H
#define DG_MY_EXCEPTIONS_H

#include <exception>
#include <string>

/*! \brief Exception for incompatible or incomplete input file information
 *
 */
class InputException : public std::exception {
  public:
    InputException(std::string msg_) : msg(msg_) {}

    const char *what() const throw() {
        return msg.c_str();
    }

  private:
    std::string msg;
};

/*! \brief Exception for non-physical results
 *
 */
class NotPhysicalException : public std::exception {
  public:
    NotPhysicalException(std::string msg_) : msg(msg_) {}

    const char * what () const throw () {
        return msg.c_str();
    }

  private:
    std::string msg;
};

/*! \brief Exception for not implemented methods
 *
 */
class NotImplementedException : public std::exception {
  public:
    NotImplementedException(std::string msg_) : msg(msg_) {}

    const char *what() const throw() {
        return msg.c_str();
    }

  private:
    std::string msg;
};

/*! \brief Exception for improper values for function
 * 
 */
class ValueErrorException : public std::exception {
public:
  ValueErrorException(std::string msg_) : msg(msg_) {}

  const char *what() const throw() {
      return msg.c_str();
  }
private:
    std::string msg;
};

/*! \brief Exception for failed system calls
 *
 */
class SystemCallException : public std::exception {
  public:
    SystemCallException(std::string msg_) : msg(msg_) {}

    const char *what() const throw() {
        return msg.c_str();
    }

  private:
    std::string msg;
};

/*! \brief Exception for fatal errors that prevent from continuing safely
 *
 */
class FatalException : public std::exception {
  public:
    FatalException(std::string msg_) : msg(msg_) {}

    const char *what() const throw() {
        return msg.c_str();
    }

  private:
    std::string msg;
};

/*! \brief Exception for failed convergence in an iterative procedure
 *
 */
class FailedConvergenceException : public std::exception {
  public:
    FailedConvergenceException(std::string msg_) : msg(msg_) {}

    const char *what() const throw() {
        return msg.c_str();
    }

  private:
    std::string msg;
};

#endif //DG_MY_EXCEPTIONS_H
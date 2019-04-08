"""
Module to handle the user inputs through console.
"""

def get_input_str(msg="Please enter",
                  only_accept=None,
                  error_msg="Please enter valid"):
    """
    Function to handle string inputs from console.

    Parameters
    ----------
    msg: str
        Input request message to display on console.
    only_accept: list
        accept only the inputs present inside this list.
    error_msg: str
        Error message to display when input is invalid

    Returns
    -------
    str
       string input by the user.
    """
    user_input = None
    while True:
        user_input = input("\n" + msg + " ->")
        if only_accept is None:
            break
        elif user_input in only_accept:
            break
        else:
            print(error_msg)
        # END OF WHILE for input
    return user_input


def get_input_int(msg="Please enter",
                  only_accept=None,
                  error_msg="Please enter valid") -> int:
    """
    Function to handle integer inputs from consoles.

    Parameters
    ----------
    msg: str
        Input request message to display on console.
    only_accept: list
        accept only the inputs present inside this list.
    error_msg: str
        Error message to display when input is invalid

    Returns
    -------
    int
       integer input by the user.
    """
    user_input = None
    while True:
        user_input = input("\n" + msg + " ->")
        try:
            user_input = int(user_input)
            if only_accept is None:
                break
            elif user_input in only_accept:
                break
            else:
                print(error_msg)
        except ValueError:
            print("Please enter a valid integer")
        # END OF WHILE for input
    return user_input


def get_confirmation(msg="Confirm",
                     error_msg="Please select y or n"):
    """
    Function to get confirmation from user through consoles.

    Parameters
    ----------
    msg: str
        Input request message to display on console.
    error_msg: str
        Error message to display when input is invalid

    Returns
    -------
    bool
       `True` if user confirms to the message, `False` otherwise.
    """
    ret_bool=None
    only_accept=['Y', 'y', 'N', 'n']
    while True:
        user_input = input("\n" + msg + " (y/n) ->")
        if user_input in only_accept:
            ret_bool = True if user_input in ['y', 'Y'] else False
            break
        else:
            print(error_msg)
        # END OF WHILE for input
    return ret_bool

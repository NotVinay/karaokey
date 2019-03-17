

def get_input_str(msg="Please enter",
                  only_accept=None,
                  error_msg="Please enter valid"):
    user_input = None
    while True:
        user_input = input("\n" + msg + " ->")
        if only_accept is None:
            break
        elif user_input in ['train', 'test']:
            break
        else:
            raise ValueError(error_msg)
        # END OF WHILE for sub_set input
    return user_input


def get_input_int(msg="Please enter",
                  only_accept=None,
                  error_msg="Please enter valid"):
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
                raise ValueError(error_msg)
        except ValueError:
            print("Please enter a valid integer")
    return user_input


def get_confirmation(msg="Confirm",
                    error_msg="Please select y or n"):
    ret_bool=None
    only_accept=['Y', 'y', 'N', 'n']
    while True:
        user_input = input("\n" + msg + " (y/n) ->")
        if user_input in ['y', 'Y', 'n', 'N']:
            ret_bool = True if user_input in ['y', 'Y'] else False
            break
        else:
            raise ValueError(error_msg)
    return ret_bool

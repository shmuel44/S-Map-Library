import os as os



def save_text_to_text_file(file_path, text):
    """Saves a string of text to a text file at the specified file path.

    Args:
        file_path (str): The file path where the text file should be saved.
        text (str): The string of text to be saved.

    Returns:
        str: The file path of the saved text file.
    """
    with open(file_path, 'w') as fp:
        fp.write(text)
    return file_path


def save_list_to_text_file(file_path, items):
    """Saves a list of items as a text file at the specified file path, with each item on a new line.

    Args:
        file_path (str): The file path where the text file should be saved.
        items (list): The list of items to be saved.

    Returns:
        str: The file path of the saved text file.
    """
    with open(file_path, 'w') as fp:
        for item in items:
            fp.write(item + '\r\n')
    return file_path

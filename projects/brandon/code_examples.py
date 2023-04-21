"""
code_examples.py

Author:
    Daniel Schonhaut
    
Dependencies: 
    Python 3.6, numpy

Description: 
    Example code for Brandon.

Last Edited: 
    11/15/21
"""
import os.path as op


def save_some_file(your_name,
                   output_dir=None):
    """Save a simple message and return the filepath to it."""
    # Get the output file name.
    if output_dir is None:
        output_dir = op.expanduser('~')
    basename = 'message_for_{}'.format(your_name)
    output_filepath = op.join(output_dir, basename)

    # Construct a message.
    msg = ("Hello {}...\n"
           "I am a meaningless file that exists\n"
           "strictly for demonstration purposes.\n"
           "Please don't pay me any more attention.\n"
           .format(your_name.title()))

    # Write message to file.
    with open(output_filepath, 'w') as file:
        file.write(msg)

    return output_filepath
